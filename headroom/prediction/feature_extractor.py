"""Comprehensive Feature Extraction System for LLM Output Length Prediction.

This module provides a complete feature extraction pipeline for predicting how long
an LLM response will be based on the input prompt. Features are organized into
five categories:

1. Text Statistics - Length metrics, vocabulary richness, compression ratio
2. Structural Features - Question patterns, lists, code blocks, formatting
3. Semantic Features - Domain detection, task type, complexity indicators
4. Embedding Features - Raw embeddings, clustering, similarity patterns
5. Meta Features - Model patterns, settings, historical data

Design Principles:
- Lazy loading for expensive dependencies (embeddings, NLP models)
- Caching for repeated computations
- Graceful degradation when optional dependencies unavailable
- Vectorized operations where possible for batch processing

Usage:
    extractor = PromptFeatureExtractor()
    features = extractor.extract(prompt)
    feature_vector = extractor.to_vector(features)

Install full dependencies:
    pip install headroom[prediction]
"""

from __future__ import annotations

import gzip
import hashlib
import logging
import math
import re
import string
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from headroom.models.config import ML_MODEL_DEFAULTS

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPE DEFINITIONS
# =============================================================================


class TaskType(str, Enum):
    """Detected task type from prompt analysis."""

    EXPLAIN = "explain"  # Explain X, What is X, How does X work
    COMPARE = "compare"  # Compare X and Y, Differences between
    GENERATE = "generate"  # Write, Create, Generate, Make
    SUMMARIZE = "summarize"  # Summarize, TL;DR, Brief overview
    ANALYZE = "analyze"  # Analyze, Evaluate, Assess
    DEBUG = "debug"  # Fix, Debug, Error, Issue
    TRANSLATE = "translate"  # Translate, Convert to
    LIST = "list"  # List, Enumerate, Give examples
    CALCULATE = "calculate"  # Calculate, Compute, Solve
    CODE = "code"  # Implement, Code, Function, Class
    EDIT = "edit"  # Edit, Modify, Update, Change
    CLASSIFY = "classify"  # Classify, Categorize, Label
    CHAT = "chat"  # Casual conversation
    INSTRUCT = "instruct"  # Step-by-step instructions
    UNKNOWN = "unknown"


class DomainType(str, Enum):
    """Detected domain/topic from prompt analysis."""

    CODE = "code"  # Programming, software
    SCIENCE = "science"  # Scientific, technical
    MATH = "math"  # Mathematical, numerical
    CREATIVE = "creative"  # Creative writing, stories
    BUSINESS = "business"  # Business, professional
    LEGAL = "legal"  # Legal, compliance
    MEDICAL = "medical"  # Medical, health
    EDUCATIONAL = "educational"  # Teaching, learning
    CONVERSATIONAL = "conversational"  # Casual chat
    FACTUAL = "factual"  # Facts, reference
    UNKNOWN = "unknown"


class ComplexityLevel(str, Enum):
    """Estimated complexity level."""

    TRIVIAL = "trivial"  # Simple lookup, yes/no
    SIMPLE = "simple"  # Single concept
    MODERATE = "moderate"  # Multiple concepts
    COMPLEX = "complex"  # Deep analysis required
    VERY_COMPLEX = "very_complex"  # Multi-step reasoning


class PromptFormat(str, Enum):
    """Detected prompt format/structure."""

    QUESTION = "question"  # Ends with ?
    INSTRUCTION = "instruction"  # Imperative command
    CONTEXT_QUERY = "context_query"  # Context + question
    MULTI_TURN = "multi_turn"  # Multiple exchanges
    TEMPLATE = "template"  # Structured template
    RAW_DATA = "raw_data"  # Data/code dump
    MIXED = "mixed"


# =============================================================================
# FEATURE DATACLASSES
# =============================================================================


@dataclass
class TextStatisticsFeatures:
    """Category 1: Text Statistics Features.

    Basic quantitative measures of the prompt text.
    These features have O(n) complexity and are fast to compute.
    """

    # Length metrics
    char_count: int = 0
    word_count: int = 0
    token_count_estimate: int = 0  # Estimated tokens (chars/4 heuristic)
    token_count_exact: int | None = None  # Exact if tokenizer available
    sentence_count: int = 0
    paragraph_count: int = 0
    line_count: int = 0

    # Average metrics
    avg_word_length: float = 0.0
    avg_sentence_length: float = 0.0  # Words per sentence
    avg_paragraph_length: float = 0.0  # Sentences per paragraph

    # Vocabulary metrics
    unique_word_count: int = 0
    vocabulary_richness: float = 0.0  # unique_words / total_words (type-token ratio)
    hapax_legomena_ratio: float = 0.0  # Words appearing exactly once / total words
    yule_k: float = 0.0  # Yule's K statistic for vocabulary richness

    # Character distribution
    uppercase_ratio: float = 0.0
    digit_ratio: float = 0.0
    punctuation_ratio: float = 0.0
    whitespace_ratio: float = 0.0
    special_char_ratio: float = 0.0

    # Compression metrics (information density)
    compression_ratio: float = 0.0  # Original / compressed size
    entropy_estimate: float = 0.0  # Shannon entropy approximation
    repetition_score: float = 0.0  # 0 = no repetition, 1 = highly repetitive

    # Readability scores (approximate)
    flesch_reading_ease: float = 0.0  # 0-100, higher = easier
    flesch_kincaid_grade: float = 0.0  # US grade level

    def to_vector(self) -> list[float]:
        """Convert to feature vector."""
        return [
            self.char_count,
            self.word_count,
            self.token_count_estimate,
            self.token_count_exact or self.token_count_estimate,
            self.sentence_count,
            self.paragraph_count,
            self.line_count,
            self.avg_word_length,
            self.avg_sentence_length,
            self.avg_paragraph_length,
            self.unique_word_count,
            self.vocabulary_richness,
            self.hapax_legomena_ratio,
            self.yule_k,
            self.uppercase_ratio,
            self.digit_ratio,
            self.punctuation_ratio,
            self.whitespace_ratio,
            self.special_char_ratio,
            self.compression_ratio,
            self.entropy_estimate,
            self.repetition_score,
            self.flesch_reading_ease,
            self.flesch_kincaid_grade,
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Get feature names for vector."""
        return [
            "char_count",
            "word_count",
            "token_count_estimate",
            "token_count_exact",
            "sentence_count",
            "paragraph_count",
            "line_count",
            "avg_word_length",
            "avg_sentence_length",
            "avg_paragraph_length",
            "unique_word_count",
            "vocabulary_richness",
            "hapax_legomena_ratio",
            "yule_k",
            "uppercase_ratio",
            "digit_ratio",
            "punctuation_ratio",
            "whitespace_ratio",
            "special_char_ratio",
            "compression_ratio",
            "entropy_estimate",
            "repetition_score",
            "flesch_reading_ease",
            "flesch_kincaid_grade",
        ]


@dataclass
class StructuralFeatures:
    """Category 2: Structural Features.

    Features derived from the structure and formatting of the prompt.
    """

    # Question patterns
    is_question: bool = False
    question_count: int = 0
    question_types: list[str] = field(default_factory=list)  # what, why, how, etc.
    has_multiple_questions: bool = False

    # List markers
    numbered_list_count: int = 0
    bullet_list_count: int = 0
    total_list_items: int = 0
    has_nested_lists: bool = False

    # Code blocks
    code_block_count: int = 0
    inline_code_count: int = 0
    code_languages_detected: list[str] = field(default_factory=list)
    total_code_lines: int = 0
    code_to_text_ratio: float = 0.0

    # Formatting markers
    header_count: int = 0  # Markdown headers
    bold_italic_count: int = 0
    link_count: int = 0
    image_reference_count: int = 0
    table_count: int = 0
    blockquote_count: int = 0

    # Delimiters and structure
    xml_tag_count: int = 0
    json_object_count: int = 0
    has_structured_template: bool = False
    delimiter_types: list[str] = field(default_factory=list)  # ---, ===, etc.

    # Conversation structure
    has_role_markers: bool = False  # User:, Assistant:, etc.
    turn_count: int = 0
    has_system_prompt_marker: bool = False

    # Special patterns
    has_examples: bool = False  # "For example", "e.g."
    example_count: int = 0
    has_constraints: bool = False  # "Must", "Should", "Don't"
    constraint_count: int = 0
    has_output_format_spec: bool = False  # Format instructions

    # Prompt engineering patterns
    has_chain_of_thought: bool = False  # "Think step by step"
    has_few_shot_examples: bool = False
    few_shot_count: int = 0
    has_persona_definition: bool = False  # "You are a..."
    has_context_window: bool = False  # Explicit context section

    def to_vector(self) -> list[float]:
        """Convert to feature vector."""
        return [
            float(self.is_question),
            self.question_count,
            len(self.question_types),
            float(self.has_multiple_questions),
            self.numbered_list_count,
            self.bullet_list_count,
            self.total_list_items,
            float(self.has_nested_lists),
            self.code_block_count,
            self.inline_code_count,
            len(self.code_languages_detected),
            self.total_code_lines,
            self.code_to_text_ratio,
            self.header_count,
            self.bold_italic_count,
            self.link_count,
            self.image_reference_count,
            self.table_count,
            self.blockquote_count,
            self.xml_tag_count,
            self.json_object_count,
            float(self.has_structured_template),
            len(self.delimiter_types),
            float(self.has_role_markers),
            self.turn_count,
            float(self.has_system_prompt_marker),
            float(self.has_examples),
            self.example_count,
            float(self.has_constraints),
            self.constraint_count,
            float(self.has_output_format_spec),
            float(self.has_chain_of_thought),
            float(self.has_few_shot_examples),
            self.few_shot_count,
            float(self.has_persona_definition),
            float(self.has_context_window),
        ]

    @classmethod
    def feature_names(cls) -> list[str]:
        """Get feature names for vector."""
        return [
            "is_question",
            "question_count",
            "question_type_count",
            "has_multiple_questions",
            "numbered_list_count",
            "bullet_list_count",
            "total_list_items",
            "has_nested_lists",
            "code_block_count",
            "inline_code_count",
            "code_language_count",
            "total_code_lines",
            "code_to_text_ratio",
            "header_count",
            "bold_italic_count",
            "link_count",
            "image_reference_count",
            "table_count",
            "blockquote_count",
            "xml_tag_count",
            "json_object_count",
            "has_structured_template",
            "delimiter_type_count",
            "has_role_markers",
            "turn_count",
            "has_system_prompt_marker",
            "has_examples",
            "example_count",
            "has_constraints",
            "constraint_count",
            "has_output_format_spec",
            "has_chain_of_thought",
            "has_few_shot_examples",
            "few_shot_count",
            "has_persona_definition",
            "has_context_window",
        ]


@dataclass
class SemanticFeatures:
    """Category 3: Semantic Features.

    Features derived from the meaning and intent of the prompt.
    """

    # Task type detection
    primary_task_type: TaskType = TaskType.UNKNOWN
    secondary_task_types: list[TaskType] = field(default_factory=list)
    task_confidence: float = 0.0

    # Domain detection
    primary_domain: DomainType = DomainType.UNKNOWN
    secondary_domains: list[DomainType] = field(default_factory=list)
    domain_confidence: float = 0.0

    # Complexity indicators
    complexity_level: ComplexityLevel = ComplexityLevel.MODERATE
    complexity_score: float = 0.5  # 0-1 continuous scale
    reasoning_depth_estimate: int = 1  # Estimated reasoning steps

    # Specificity
    specificity_score: float = 0.5  # 0 = vague, 1 = very specific
    has_specific_entities: bool = False
    named_entity_count: int = 0
    named_entity_types: list[str] = field(default_factory=list)

    # Intent signals
    requires_factual_recall: bool = False
    requires_reasoning: bool = False
    requires_creativity: bool = False
    requires_code_generation: bool = False
    requires_structured_output: bool = False

    # Output length hints (explicit)
    explicit_length_request: str | None = None  # "brief", "detailed", "100 words"
    requested_word_count: int | None = None
    requested_paragraph_count: int | None = None
    requested_item_count: int | None = None  # For lists

    # Sentiment and tone
    prompt_sentiment: str = "neutral"  # positive, negative, neutral
    formality_level: float = 0.5  # 0 = casual, 1 = formal
    urgency_indicators: int = 0  # ASAP, urgent, quickly

    # Topic keywords
    top_keywords: list[str] = field(default_factory=list)
    keyword_density: float = 0.0

    # Format specification
    prompt_format: PromptFormat = PromptFormat.INSTRUCTION

    def to_vector(self) -> list[float]:
        """Convert to feature vector."""
        task_type_encoding = [0.0] * len(TaskType)
        if self.primary_task_type != TaskType.UNKNOWN:
            task_type_encoding[list(TaskType).index(self.primary_task_type)] = 1.0

        domain_encoding = [0.0] * len(DomainType)
        if self.primary_domain != DomainType.UNKNOWN:
            domain_encoding[list(DomainType).index(self.primary_domain)] = 1.0

        complexity_encoding = [0.0] * len(ComplexityLevel)
        complexity_encoding[list(ComplexityLevel).index(self.complexity_level)] = 1.0

        format_encoding = [0.0] * len(PromptFormat)
        format_encoding[list(PromptFormat).index(self.prompt_format)] = 1.0

        return (
            task_type_encoding
            + [self.task_confidence]
            + domain_encoding
            + [self.domain_confidence]
            + complexity_encoding
            + [
                self.complexity_score,
                self.reasoning_depth_estimate,
                self.specificity_score,
                float(self.has_specific_entities),
                self.named_entity_count,
                len(self.named_entity_types),
                float(self.requires_factual_recall),
                float(self.requires_reasoning),
                float(self.requires_creativity),
                float(self.requires_code_generation),
                float(self.requires_structured_output),
                1.0 if self.explicit_length_request else 0.0,
                self.requested_word_count or 0,
                self.requested_paragraph_count or 0,
                self.requested_item_count or 0,
                1.0
                if self.prompt_sentiment == "positive"
                else (-1.0 if self.prompt_sentiment == "negative" else 0.0),
                self.formality_level,
                self.urgency_indicators,
                len(self.top_keywords),
                self.keyword_density,
            ]
            + format_encoding
        )

    @classmethod
    def feature_names(cls) -> list[str]:
        """Get feature names for vector."""
        task_names = [f"task_type_{t.value}" for t in TaskType]
        domain_names = [f"domain_{d.value}" for d in DomainType]
        complexity_names = [f"complexity_{c.value}" for c in ComplexityLevel]
        format_names = [f"format_{f.value}" for f in PromptFormat]

        return (
            task_names
            + ["task_confidence"]
            + domain_names
            + ["domain_confidence"]
            + complexity_names
            + [
                "complexity_score",
                "reasoning_depth_estimate",
                "specificity_score",
                "has_specific_entities",
                "named_entity_count",
                "named_entity_type_count",
                "requires_factual_recall",
                "requires_reasoning",
                "requires_creativity",
                "requires_code_generation",
                "requires_structured_output",
                "has_explicit_length_request",
                "requested_word_count",
                "requested_paragraph_count",
                "requested_item_count",
                "sentiment_score",
                "formality_level",
                "urgency_indicators",
                "keyword_count",
                "keyword_density",
            ]
            + format_names
        )


@dataclass
class EmbeddingFeatures:
    """Category 4: Embedding-based Features.

    Features derived from neural embeddings of the prompt.
    These require sentence-transformers or similar models.
    """

    # Raw embedding (optional, for downstream use)
    raw_embedding: list[float] | None = None
    embedding_dim: int = 0

    # Embedding statistics
    embedding_norm: float = 0.0
    embedding_mean: float = 0.0
    embedding_std: float = 0.0
    embedding_max: float = 0.0
    embedding_min: float = 0.0

    # Similarity to known patterns
    similarity_to_short_response_cluster: float = 0.0
    similarity_to_long_response_cluster: float = 0.0
    similarity_to_code_cluster: float = 0.0
    similarity_to_explanation_cluster: float = 0.0
    similarity_to_list_cluster: float = 0.0

    # Clustering features
    predicted_cluster_id: int = -1
    cluster_confidence: float = 0.0
    distance_to_cluster_center: float = 0.0

    # Semantic density
    embedding_entropy: float = 0.0  # Entropy of embedding values
    information_content_score: float = 0.0

    # Cross-attention features (if available)
    attention_concentration: float = 0.0
    attention_spread: float = 0.0

    def to_vector(self, include_raw: bool = False) -> list[float]:
        """Convert to feature vector.

        Args:
            include_raw: If True, include raw embedding (can be large).
        """
        features = [
            self.embedding_dim,
            self.embedding_norm,
            self.embedding_mean,
            self.embedding_std,
            self.embedding_max,
            self.embedding_min,
            self.similarity_to_short_response_cluster,
            self.similarity_to_long_response_cluster,
            self.similarity_to_code_cluster,
            self.similarity_to_explanation_cluster,
            self.similarity_to_list_cluster,
            self.predicted_cluster_id,
            self.cluster_confidence,
            self.distance_to_cluster_center,
            self.embedding_entropy,
            self.information_content_score,
            self.attention_concentration,
            self.attention_spread,
        ]

        if include_raw and self.raw_embedding:
            features.extend(self.raw_embedding)

        return features

    @classmethod
    def feature_names(cls, include_raw: bool = False, embedding_dim: int = 0) -> list[str]:
        """Get feature names for vector."""
        names = [
            "embedding_dim",
            "embedding_norm",
            "embedding_mean",
            "embedding_std",
            "embedding_max",
            "embedding_min",
            "sim_short_response_cluster",
            "sim_long_response_cluster",
            "sim_code_cluster",
            "sim_explanation_cluster",
            "sim_list_cluster",
            "predicted_cluster_id",
            "cluster_confidence",
            "distance_to_cluster_center",
            "embedding_entropy",
            "information_content_score",
            "attention_concentration",
            "attention_spread",
        ]

        if include_raw:
            names.extend([f"embedding_{i}" for i in range(embedding_dim)])

        return names


@dataclass
class MetaFeatures:
    """Category 5: Meta Features.

    Features related to model, settings, and historical patterns.
    """

    # Model information
    model_name: str = ""
    model_family: str = ""  # gpt, claude, llama, etc.
    model_size_category: str = ""  # small, medium, large, xl
    model_context_limit: int = 0

    # Generation settings (if known)
    temperature: float | None = None
    max_tokens_setting: int | None = None
    top_p: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    # Context utilization
    prompt_context_ratio: float = 0.0  # prompt_tokens / context_limit
    available_output_tokens: int = 0

    # Historical patterns (if available)
    user_avg_response_length: float | None = None
    similar_prompt_avg_response: float | None = None
    historical_response_variance: float | None = None

    # Prompt hash for lookup
    prompt_hash: str = ""
    prompt_signature: str = ""  # Simplified hash of structure

    # Time features
    is_first_turn: bool = True
    conversation_turn_number: int = 0
    cumulative_context_tokens: int = 0

    # System prompt features
    system_prompt_length: int = 0
    system_prompt_token_estimate: int = 0
    has_output_constraints_in_system: bool = False

    def to_vector(self) -> list[float]:
        """Convert to feature vector."""
        # Encode model family
        model_families = ["gpt", "claude", "llama", "mistral", "gemini", "other"]
        family_encoding = [0.0] * len(model_families)
        family_lower = self.model_family.lower()
        for i, family in enumerate(model_families):
            if family in family_lower:
                family_encoding[i] = 1.0
                break
        else:
            family_encoding[-1] = 1.0  # "other"

        # Encode model size
        sizes = ["small", "medium", "large", "xl"]
        size_encoding = [0.0] * len(sizes)
        size_lower = self.model_size_category.lower()
        for i, size in enumerate(sizes):
            if size in size_lower:
                size_encoding[i] = 1.0
                break

        return (
            family_encoding
            + size_encoding
            + [
                self.model_context_limit,
                self.temperature if self.temperature is not None else 0.7,
                self.max_tokens_setting if self.max_tokens_setting is not None else 0,
                self.top_p if self.top_p is not None else 1.0,
                self.presence_penalty if self.presence_penalty is not None else 0.0,
                self.frequency_penalty if self.frequency_penalty is not None else 0.0,
                self.prompt_context_ratio,
                self.available_output_tokens,
                self.user_avg_response_length if self.user_avg_response_length is not None else 0,
                self.similar_prompt_avg_response
                if self.similar_prompt_avg_response is not None
                else 0,
                self.historical_response_variance
                if self.historical_response_variance is not None
                else 0,
                float(self.is_first_turn),
                self.conversation_turn_number,
                self.cumulative_context_tokens,
                self.system_prompt_length,
                self.system_prompt_token_estimate,
                float(self.has_output_constraints_in_system),
            ]
        )

    @classmethod
    def feature_names(cls) -> list[str]:
        """Get feature names for vector."""
        model_families = ["gpt", "claude", "llama", "mistral", "gemini", "other"]
        family_names = [f"model_family_{f}" for f in model_families]
        sizes = ["small", "medium", "large", "xl"]
        size_names = [f"model_size_{s}" for s in sizes]

        return (
            family_names
            + size_names
            + [
                "model_context_limit",
                "temperature",
                "max_tokens_setting",
                "top_p",
                "presence_penalty",
                "frequency_penalty",
                "prompt_context_ratio",
                "available_output_tokens",
                "user_avg_response_length",
                "similar_prompt_avg_response",
                "historical_response_variance",
                "is_first_turn",
                "conversation_turn_number",
                "cumulative_context_tokens",
                "system_prompt_length",
                "system_prompt_token_estimate",
                "has_output_constraints_in_system",
            ]
        )


@dataclass
class PromptFeatures:
    """Complete feature set for a prompt."""

    text_statistics: TextStatisticsFeatures = field(default_factory=TextStatisticsFeatures)
    structural: StructuralFeatures = field(default_factory=StructuralFeatures)
    semantic: SemanticFeatures = field(default_factory=SemanticFeatures)
    embedding: EmbeddingFeatures = field(default_factory=EmbeddingFeatures)
    meta: MetaFeatures = field(default_factory=MetaFeatures)

    # Original prompt for reference
    original_prompt: str = ""
    extraction_timestamp: str = ""

    def to_vector(self, include_raw_embedding: bool = False) -> list[float]:
        """Convert all features to a single vector."""
        return (
            self.text_statistics.to_vector()
            + self.structural.to_vector()
            + self.semantic.to_vector()
            + self.embedding.to_vector(include_raw=include_raw_embedding)
            + self.meta.to_vector()
        )

    @classmethod
    def feature_names(
        cls, include_raw_embedding: bool = False, embedding_dim: int = 384
    ) -> list[str]:
        """Get all feature names."""
        return (
            TextStatisticsFeatures.feature_names()
            + StructuralFeatures.feature_names()
            + SemanticFeatures.feature_names()
            + EmbeddingFeatures.feature_names(
                include_raw=include_raw_embedding, embedding_dim=embedding_dim
            )
            + MetaFeatures.feature_names()
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text_statistics": {
                k: v for k, v in self.text_statistics.__dict__.items() if not k.startswith("_")
            },
            "structural": {
                k: v if not isinstance(v, list) else v
                for k, v in self.structural.__dict__.items()
                if not k.startswith("_")
            },
            "semantic": {
                k: (v.value if isinstance(v, Enum) else v)
                for k, v in self.semantic.__dict__.items()
                if not k.startswith("_")
            },
            "embedding": {
                k: v
                for k, v in self.embedding.__dict__.items()
                if not k.startswith("_") and k != "raw_embedding"
            },
            "meta": {k: v for k, v in self.meta.__dict__.items() if not k.startswith("_")},
        }


# =============================================================================
# FEATURE EXTRACTORS (Individual Components)
# =============================================================================


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors."""

    @abstractmethod
    def extract(self, text: str, **kwargs: Any) -> Any:
        """Extract features from text."""
        ...


class TextStatisticsExtractor(BaseFeatureExtractor):
    """Extracts text statistics features."""

    # Sentence ending patterns
    SENTENCE_ENDINGS = re.compile(r"[.!?]+")
    PARAGRAPH_PATTERN = re.compile(r"\n\s*\n")

    # Syllable counting approximation
    VOWELS = set("aeiouyAEIOUY")

    def __init__(self, tokenizer: Any | None = None):
        """Initialize with optional tokenizer for exact token counts.

        Args:
            tokenizer: Optional tokenizer with count_text(str) -> int method.
        """
        self.tokenizer = tokenizer

    def extract(self, text: str, **kwargs: Any) -> TextStatisticsFeatures:
        """Extract text statistics features.

        Args:
            text: Input text to analyze.

        Returns:
            TextStatisticsFeatures dataclass.
        """
        if not text or not text.strip():
            return TextStatisticsFeatures()

        features = TextStatisticsFeatures()

        # Length metrics
        features.char_count = len(text)
        words = text.split()
        features.word_count = len(words)
        features.token_count_estimate = features.char_count // 4
        features.line_count = text.count("\n") + 1

        # Exact token count if tokenizer available
        if self.tokenizer is not None:
            try:
                features.token_count_exact = self.tokenizer.count_text(text)
            except Exception as e:
                logger.debug(f"Tokenizer failed: {e}")

        # Sentences and paragraphs
        sentences = [s.strip() for s in self.SENTENCE_ENDINGS.split(text) if s.strip()]
        features.sentence_count = max(1, len(sentences))
        paragraphs = [p.strip() for p in self.PARAGRAPH_PATTERN.split(text) if p.strip()]
        features.paragraph_count = max(1, len(paragraphs))

        # Average metrics
        if features.word_count > 0:
            features.avg_word_length = sum(len(w) for w in words) / features.word_count
        if features.sentence_count > 0:
            features.avg_sentence_length = features.word_count / features.sentence_count
        if features.paragraph_count > 0:
            features.avg_paragraph_length = features.sentence_count / features.paragraph_count

        # Vocabulary metrics
        words_lower = [w.lower() for w in words]
        word_freq = Counter(words_lower)
        features.unique_word_count = len(word_freq)

        if features.word_count > 0:
            features.vocabulary_richness = features.unique_word_count / features.word_count
            # Hapax legomena (words appearing once)
            hapax_count = sum(1 for count in word_freq.values() if count == 1)
            features.hapax_legomena_ratio = hapax_count / features.word_count
            # Yule's K
            features.yule_k = self._calculate_yule_k(word_freq)

        # Character distribution
        if features.char_count > 0:
            features.uppercase_ratio = sum(1 for c in text if c.isupper()) / features.char_count
            features.digit_ratio = sum(1 for c in text if c.isdigit()) / features.char_count
            features.punctuation_ratio = (
                sum(1 for c in text if c in string.punctuation) / features.char_count
            )
            features.whitespace_ratio = sum(1 for c in text if c.isspace()) / features.char_count
            special_chars = set(text) - set(
                string.ascii_letters + string.digits + string.whitespace
            )
            features.special_char_ratio = (
                sum(1 for c in text if c in special_chars) / features.char_count
            )

        # Compression metrics
        features.compression_ratio = self._calculate_compression_ratio(text)
        features.entropy_estimate = self._calculate_entropy(text)
        features.repetition_score = self._calculate_repetition_score(text)

        # Readability
        syllable_count = self._count_syllables(text)
        if features.sentence_count > 0 and features.word_count > 0:
            features.flesch_reading_ease = self._flesch_reading_ease(
                features.word_count, features.sentence_count, syllable_count
            )
            features.flesch_kincaid_grade = self._flesch_kincaid_grade(
                features.word_count, features.sentence_count, syllable_count
            )

        return features

    def _calculate_yule_k(self, word_freq: Counter) -> float:
        """Calculate Yule's K statistic for vocabulary richness."""
        n = sum(word_freq.values())
        if n <= 1:
            return 0.0

        freq_of_freq = Counter(word_freq.values())
        m1 = n
        m2 = sum(freq * (count**2) for freq, count in freq_of_freq.items())

        if m1 == 0:
            return 0.0

        k = 10000 * (m2 - m1) / (m1 * m1)
        return max(0.0, k)

    def _calculate_compression_ratio(self, text: str) -> float:
        """Calculate compression ratio using gzip."""
        if not text:
            return 0.0
        try:
            original = text.encode("utf-8")
            compressed = gzip.compress(original)
            return len(original) / max(1, len(compressed))
        except Exception:
            return 1.0

    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0

        freq = Counter(text)
        total = len(text)
        entropy = 0.0

        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def _calculate_repetition_score(self, text: str) -> float:
        """Calculate repetition score (0 = unique, 1 = highly repetitive)."""
        if not text or len(text) < 10:
            return 0.0

        # Use n-gram repetition
        n = 3
        ngrams = [text[i : i + n] for i in range(len(text) - n + 1)]
        if not ngrams:
            return 0.0

        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)

        # Inverse uniqueness ratio
        return 1.0 - (unique_ngrams / total_ngrams)

    def _count_syllables(self, text: str) -> int:
        """Approximate syllable count."""
        words = text.lower().split()
        total = 0

        for word in words:
            word = "".join(c for c in word if c.isalpha())
            if not word:
                continue

            # Count vowel groups
            syllables = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in self.VOWELS
                if is_vowel and not prev_vowel:
                    syllables += 1
                prev_vowel = is_vowel

            # Handle silent e
            if word.endswith("e"):
                syllables = max(1, syllables - 1)

            total += max(1, syllables)

        return total

    def _flesch_reading_ease(self, words: int, sentences: int, syllables: int) -> float:
        """Calculate Flesch Reading Ease score."""
        if sentences == 0 or words == 0:
            return 0.0
        score = 206.835 - 1.015 * (words / sentences) - 84.6 * (syllables / words)
        return max(0.0, min(100.0, score))

    def _flesch_kincaid_grade(self, words: int, sentences: int, syllables: int) -> float:
        """Calculate Flesch-Kincaid Grade Level."""
        if sentences == 0 or words == 0:
            return 0.0
        grade = 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59
        return max(0.0, grade)


class StructuralExtractor(BaseFeatureExtractor):
    """Extracts structural features from text."""

    # Regex patterns
    QUESTION_PATTERN = re.compile(r"\?")
    QUESTION_WORDS = re.compile(
        r"\b(what|why|how|when|where|who|which|whose|whom|can|could|would|should|is|are|do|does|did)\b",
        re.IGNORECASE,
    )
    NUMBERED_LIST = re.compile(r"^\s*\d+[\.\)]\s+", re.MULTILINE)
    BULLET_LIST = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
    CODE_BLOCK = re.compile(r"```(\w*)\n[\s\S]*?```")
    INLINE_CODE = re.compile(r"`[^`]+`")
    MARKDOWN_HEADER = re.compile(r"^#+\s+", re.MULTILINE)
    BOLD_ITALIC = re.compile(r"\*\*[^*]+\*\*|\*[^*]+\*|__[^_]+__|_[^_]+_")
    LINK_PATTERN = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
    IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
    TABLE_PATTERN = re.compile(r"\|[^|]+\|")
    BLOCKQUOTE = re.compile(r"^\s*>\s+", re.MULTILINE)
    XML_TAG = re.compile(r"<[^>]+>")
    JSON_OBJECT = re.compile(r"\{[^{}]*\}")
    DELIMITER = re.compile(r"^[-=_]{3,}$", re.MULTILINE)
    ROLE_MARKER = re.compile(
        r"^(User|Assistant|System|Human|AI|Bot):\s*", re.MULTILINE | re.IGNORECASE
    )
    EXAMPLE_PATTERN = re.compile(
        r"\b(for example|e\.g\.|example[s]?:|such as|like this)\b", re.IGNORECASE
    )
    CONSTRAINT_PATTERN = re.compile(
        r"\b(must|should|don't|do not|cannot|can't|never|always|required|necessary)\b",
        re.IGNORECASE,
    )
    COT_PATTERN = re.compile(
        r"\b(step by step|think through|let's think|reasoning|chain of thought)\b",
        re.IGNORECASE,
    )
    PERSONA_PATTERN = re.compile(r"\b(you are|act as|pretend to be|role of)\b", re.IGNORECASE)
    CONTEXT_PATTERN = re.compile(
        r"\b(context|background|given|provided|following information)\b", re.IGNORECASE
    )
    OUTPUT_FORMAT_PATTERN = re.compile(
        r"\b(format|output|respond|answer)\s+(as|in|with|using)\b", re.IGNORECASE
    )

    # Language detection for code blocks
    CODE_LANGUAGES = {
        "python",
        "javascript",
        "typescript",
        "java",
        "c",
        "cpp",
        "csharp",
        "go",
        "rust",
        "ruby",
        "php",
        "swift",
        "kotlin",
        "scala",
        "shell",
        "bash",
        "sql",
        "html",
        "css",
        "json",
        "yaml",
        "xml",
        "markdown",
    }

    def extract(self, text: str, **kwargs: Any) -> StructuralFeatures:
        """Extract structural features from text."""
        if not text:
            return StructuralFeatures()

        features = StructuralFeatures()

        # Question detection
        questions = self.QUESTION_PATTERN.findall(text)
        features.question_count = len(questions)
        features.is_question = features.question_count > 0
        features.has_multiple_questions = features.question_count > 1

        # Question types
        question_words = self.QUESTION_WORDS.findall(text.lower())
        features.question_types = list(set(question_words))

        # List markers
        numbered_matches = self.NUMBERED_LIST.findall(text)
        features.numbered_list_count = len(numbered_matches)

        bullet_matches = self.BULLET_LIST.findall(text)
        features.bullet_list_count = len(bullet_matches)

        features.total_list_items = features.numbered_list_count + features.bullet_list_count

        # Check for nested lists (indented list items)
        nested_pattern = re.compile(r"^\s{2,}[-*+\d]", re.MULTILINE)
        features.has_nested_lists = bool(nested_pattern.search(text))

        # Code blocks
        code_blocks = self.CODE_BLOCK.findall(text)
        features.code_block_count = len(self.CODE_BLOCK.findall(text))

        # Extract languages from code blocks
        languages = [lang.lower() for lang in code_blocks if lang]
        features.code_languages_detected = [
            lang for lang in languages if lang in self.CODE_LANGUAGES
        ]

        # Count code lines
        for match in self.CODE_BLOCK.finditer(text):
            block_content = match.group(0)
            features.total_code_lines += block_content.count("\n")

        # Inline code
        features.inline_code_count = len(self.INLINE_CODE.findall(text))

        # Code to text ratio
        total_code_chars = sum(len(m.group(0)) for m in self.CODE_BLOCK.finditer(text)) + sum(
            len(m.group(0)) for m in self.INLINE_CODE.finditer(text)
        )
        if len(text) > 0:
            features.code_to_text_ratio = total_code_chars / len(text)

        # Formatting markers
        features.header_count = len(self.MARKDOWN_HEADER.findall(text))
        features.bold_italic_count = len(self.BOLD_ITALIC.findall(text))
        features.link_count = len(self.LINK_PATTERN.findall(text))
        features.image_reference_count = len(self.IMAGE_PATTERN.findall(text))
        features.table_count = len(self.TABLE_PATTERN.findall(text)) // 2  # Approximate rows
        features.blockquote_count = len(self.BLOCKQUOTE.findall(text))

        # Structure detection
        features.xml_tag_count = len(self.XML_TAG.findall(text))
        features.json_object_count = len(self.JSON_OBJECT.findall(text))

        delimiters = self.DELIMITER.findall(text)
        features.delimiter_types = list({d[0] for d in delimiters if d})
        features.has_structured_template = (
            features.xml_tag_count > 2
            or bool(features.delimiter_types)
            or features.json_object_count > 0
        )

        # Conversation structure
        role_markers = self.ROLE_MARKER.findall(text)
        features.has_role_markers = len(role_markers) > 0
        features.turn_count = len(role_markers)
        features.has_system_prompt_marker = any("system" in m.lower() for m in role_markers)

        # Examples and constraints
        example_matches = self.EXAMPLE_PATTERN.findall(text)
        features.has_examples = len(example_matches) > 0
        features.example_count = len(example_matches)

        constraint_matches = self.CONSTRAINT_PATTERN.findall(text)
        features.has_constraints = len(constraint_matches) > 0
        features.constraint_count = len(constraint_matches)

        # Output format specification
        features.has_output_format_spec = bool(self.OUTPUT_FORMAT_PATTERN.search(text))

        # Prompt engineering patterns
        features.has_chain_of_thought = bool(self.COT_PATTERN.search(text))
        features.has_persona_definition = bool(self.PERSONA_PATTERN.search(text))
        features.has_context_window = bool(self.CONTEXT_PATTERN.search(text))

        # Few-shot detection (multiple examples with consistent structure)
        if features.example_count >= 2:
            features.has_few_shot_examples = True
            features.few_shot_count = features.example_count

        return features


class SemanticExtractor(BaseFeatureExtractor):
    """Extracts semantic features from text.

    Uses keyword-based detection and optional NLP models.
    """

    # Task type keywords
    TASK_KEYWORDS: ClassVar[dict[TaskType, list[str]]] = {
        TaskType.EXPLAIN: [
            "explain",
            "what is",
            "what are",
            "how does",
            "how do",
            "describe",
            "define",
            "clarify",
            "elaborate",
            "tell me about",
        ],
        TaskType.COMPARE: [
            "compare",
            "contrast",
            "difference",
            "differences",
            "versus",
            "vs",
            "better",
            "worse",
            "similar",
            "distinction",
        ],
        TaskType.GENERATE: [
            "write",
            "create",
            "generate",
            "make",
            "compose",
            "draft",
            "produce",
            "design",
            "build",
        ],
        TaskType.SUMMARIZE: [
            "summarize",
            "summary",
            "tldr",
            "brief",
            "overview",
            "condense",
            "shorten",
            "recap",
            "main points",
        ],
        TaskType.ANALYZE: [
            "analyze",
            "analyse",
            "evaluate",
            "assess",
            "examine",
            "investigate",
            "review",
            "critique",
            "study",
        ],
        TaskType.DEBUG: [
            "fix",
            "debug",
            "error",
            "bug",
            "issue",
            "problem",
            "wrong",
            "broken",
            "not working",
            "fails",
        ],
        TaskType.TRANSLATE: [
            "translate",
            "convert",
            "transform",
            "change to",
            "in french",
            "in spanish",
            "to english",
        ],
        TaskType.LIST: [
            "list",
            "enumerate",
            "give examples",
            "name",
            "provide",
            "what are some",
            "top",
            "best",
        ],
        TaskType.CALCULATE: [
            "calculate",
            "compute",
            "solve",
            "find the",
            "what is the value",
            "how much",
            "how many",
        ],
        TaskType.CODE: [
            "implement",
            "code",
            "function",
            "class",
            "program",
            "script",
            "algorithm",
            "method",
            "api",
        ],
        TaskType.EDIT: [
            "edit",
            "modify",
            "update",
            "change",
            "revise",
            "improve",
            "rewrite",
            "refactor",
            "correct",
        ],
        TaskType.CLASSIFY: [
            "classify",
            "categorize",
            "label",
            "identify",
            "determine",
            "which type",
            "what kind",
        ],
        TaskType.CHAT: [
            "hi",
            "hello",
            "hey",
            "thanks",
            "thank you",
            "how are you",
            "nice",
            "cool",
            "okay",
        ],
        TaskType.INSTRUCT: [
            "steps",
            "step-by-step",
            "how to",
            "guide",
            "tutorial",
            "instructions",
            "procedure",
            "process",
        ],
    }

    # Domain keywords
    DOMAIN_KEYWORDS: ClassVar[dict[DomainType, list[str]]] = {
        DomainType.CODE: [
            "code",
            "programming",
            "function",
            "variable",
            "class",
            "api",
            "database",
            "software",
            "developer",
            "python",
            "javascript",
            "algorithm",
        ],
        DomainType.SCIENCE: [
            "science",
            "scientific",
            "research",
            "experiment",
            "hypothesis",
            "theory",
            "physics",
            "chemistry",
            "biology",
            "study",
        ],
        DomainType.MATH: [
            "math",
            "mathematics",
            "equation",
            "formula",
            "calculate",
            "number",
            "algebra",
            "geometry",
            "calculus",
            "statistic",
        ],
        DomainType.CREATIVE: [
            "story",
            "poem",
            "creative",
            "fiction",
            "character",
            "narrative",
            "write",
            "imagine",
            "fantasy",
            "novel",
        ],
        DomainType.BUSINESS: [
            "business",
            "company",
            "market",
            "finance",
            "investment",
            "strategy",
            "management",
            "profit",
            "revenue",
            "customer",
        ],
        DomainType.LEGAL: [
            "legal",
            "law",
            "court",
            "contract",
            "attorney",
            "lawyer",
            "regulation",
            "compliance",
            "rights",
            "liability",
        ],
        DomainType.MEDICAL: [
            "medical",
            "health",
            "doctor",
            "patient",
            "disease",
            "treatment",
            "symptom",
            "diagnosis",
            "medicine",
            "hospital",
        ],
        DomainType.EDUCATIONAL: [
            "learn",
            "teach",
            "education",
            "student",
            "school",
            "course",
            "lesson",
            "study",
            "training",
            "curriculum",
        ],
        DomainType.CONVERSATIONAL: [
            "chat",
            "talk",
            "conversation",
            "discuss",
            "opinion",
            "think",
            "feel",
            "casual",
        ],
        DomainType.FACTUAL: [
            "fact",
            "information",
            "data",
            "statistic",
            "history",
            "event",
            "date",
            "when",
            "where",
            "who",
        ],
    }

    # Length request patterns
    LENGTH_PATTERNS: ClassVar[list[tuple[re.Pattern, str]]] = [
        (re.compile(r"\b(\d+)\s*words?\b", re.IGNORECASE), "words"),
        (re.compile(r"\b(\d+)\s*paragraphs?\b", re.IGNORECASE), "paragraphs"),
        (re.compile(r"\b(\d+)\s*sentences?\b", re.IGNORECASE), "sentences"),
        (re.compile(r"\b(\d+)\s*items?\b", re.IGNORECASE), "items"),
        (re.compile(r"\b(\d+)\s*points?\b", re.IGNORECASE), "items"),
        (re.compile(r"\bbrief(?:ly)?\b", re.IGNORECASE), "brief"),
        (re.compile(r"\bshort(?:ly)?\b", re.IGNORECASE), "short"),
        (re.compile(r"\bdetailed\b", re.IGNORECASE), "detailed"),
        (re.compile(r"\bcomprehensive\b", re.IGNORECASE), "comprehensive"),
        (re.compile(r"\bin[-\s]?depth\b", re.IGNORECASE), "detailed"),
        (re.compile(r"\bconcise(?:ly)?\b", re.IGNORECASE), "brief"),
        (re.compile(r"\bthorough(?:ly)?\b", re.IGNORECASE), "detailed"),
    ]

    # Sentiment words
    POSITIVE_WORDS = frozenset(
        ["good", "great", "excellent", "amazing", "wonderful", "fantastic", "love", "like", "best"]
    )
    NEGATIVE_WORDS = frozenset(
        ["bad", "terrible", "awful", "horrible", "hate", "worst", "poor", "wrong", "fail"]
    )

    # Urgency indicators
    URGENCY_WORDS = frozenset(
        ["urgent", "asap", "immediately", "quickly", "fast", "now", "hurry", "rush", "critical"]
    )

    def __init__(self, use_ner: bool = False):
        """Initialize semantic extractor.

        Args:
            use_ner: If True, use spaCy for named entity recognition (slower).
        """
        self.use_ner = use_ner
        self._nlp = None  # Lazy load

    def extract(self, text: str, **kwargs: Any) -> SemanticFeatures:
        """Extract semantic features from text."""
        if not text:
            return SemanticFeatures()

        features = SemanticFeatures()
        text_lower = text.lower()
        words = text_lower.split()

        # Task type detection
        task_scores = self._detect_task_type(text_lower)
        if task_scores:
            best_task = max(task_scores.items(), key=lambda x: x[1])
            features.primary_task_type = best_task[0]
            features.task_confidence = best_task[1]

            # Secondary tasks (confidence > 0.3)
            features.secondary_task_types = [
                task
                for task, score in task_scores.items()
                if score > 0.3 and task != features.primary_task_type
            ]

        # Domain detection
        domain_scores = self._detect_domain(text_lower)
        if domain_scores:
            best_domain = max(domain_scores.items(), key=lambda x: x[1])
            features.primary_domain = best_domain[0]
            features.domain_confidence = best_domain[1]

            features.secondary_domains = [
                domain
                for domain, score in domain_scores.items()
                if score > 0.3 and domain != features.primary_domain
            ]

        # Complexity estimation
        features.complexity_level, features.complexity_score = self._estimate_complexity(
            text, features.primary_task_type
        )
        features.reasoning_depth_estimate = self._estimate_reasoning_depth(text)

        # Specificity
        features.specificity_score = self._calculate_specificity(text)

        # Named entities (if NER enabled)
        if self.use_ner:
            entities = self._extract_entities(text)
            features.has_specific_entities = len(entities) > 0
            features.named_entity_count = len(entities)
            features.named_entity_types = list({e[1] for e in entities})

        # Intent signals
        features.requires_factual_recall = self._check_factual_recall(text_lower)
        features.requires_reasoning = self._check_reasoning(text_lower)
        features.requires_creativity = self._check_creativity(text_lower)
        features.requires_code_generation = features.primary_task_type == TaskType.CODE or (
            features.primary_domain == DomainType.CODE and "write" in text_lower
        )
        features.requires_structured_output = self._check_structured_output(text_lower)

        # Length requests
        length_info = self._extract_length_request(text)
        features.explicit_length_request = length_info.get("type")
        features.requested_word_count = length_info.get("words")
        features.requested_paragraph_count = length_info.get("paragraphs")
        features.requested_item_count = length_info.get("items")

        # Sentiment
        features.prompt_sentiment = self._detect_sentiment(words)

        # Formality
        features.formality_level = self._estimate_formality(text)

        # Urgency
        features.urgency_indicators = sum(1 for w in words if w in self.URGENCY_WORDS)

        # Keywords
        features.top_keywords = self._extract_keywords(text, n=10)
        if len(words) > 0:
            features.keyword_density = len(features.top_keywords) / len(words)

        # Prompt format
        features.prompt_format = self._detect_format(text)

        return features

    def _detect_task_type(self, text: str) -> dict[TaskType, float]:
        """Detect task type from keywords."""
        scores: dict[TaskType, float] = {}

        for task_type, keywords in self.TASK_KEYWORDS.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text:
                    # Weight by position (earlier = stronger signal)
                    pos = text.find(keyword)
                    position_weight = 1.0 - (pos / max(1, len(text))) * 0.5
                    score += position_weight

            if score > 0:
                # Normalize by number of keywords
                scores[task_type] = min(1.0, score / len(keywords) * 2)

        return scores

    def _detect_domain(self, text: str) -> dict[DomainType, float]:
        """Detect domain from keywords."""
        scores: dict[DomainType, float] = {}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[domain] = min(1.0, score / len(keywords) * 3)

        return scores

    def _estimate_complexity(self, text: str, task_type: TaskType) -> tuple[ComplexityLevel, float]:
        """Estimate prompt complexity."""
        score = 0.5  # Base

        # Length factor
        word_count = len(text.split())
        if word_count < 10:
            score -= 0.2
        elif word_count > 100:
            score += 0.2
        elif word_count > 500:
            score += 0.3

        # Question complexity
        question_count = text.count("?")
        if question_count > 3:
            score += 0.2

        # Multi-part requests
        if re.search(r"\b(and|also|additionally|furthermore)\b", text, re.IGNORECASE):
            score += 0.1

        # Task-based adjustment
        complex_tasks = {TaskType.ANALYZE, TaskType.COMPARE, TaskType.DEBUG}
        if task_type in complex_tasks:
            score += 0.15

        simple_tasks = {TaskType.CHAT, TaskType.LIST}
        if task_type in simple_tasks:
            score -= 0.15

        # Clamp score
        score = max(0.0, min(1.0, score))

        # Map to level
        if score < 0.2:
            level = ComplexityLevel.TRIVIAL
        elif score < 0.4:
            level = ComplexityLevel.SIMPLE
        elif score < 0.6:
            level = ComplexityLevel.MODERATE
        elif score < 0.8:
            level = ComplexityLevel.COMPLEX
        else:
            level = ComplexityLevel.VERY_COMPLEX

        return level, score

    def _estimate_reasoning_depth(self, text: str) -> int:
        """Estimate number of reasoning steps required."""
        depth = 1

        # Multi-step indicators
        step_indicators = [
            "first",
            "then",
            "next",
            "finally",
            "step",
            "after that",
            "before",
            "because",
            "therefore",
            "thus",
            "hence",
        ]
        for indicator in step_indicators:
            if indicator in text.lower():
                depth += 1

        # Question depth
        depth += min(3, text.count("?") - 1)

        return max(1, min(10, depth))

    def _calculate_specificity(self, text: str) -> float:
        """Calculate how specific vs vague the prompt is."""
        specificity = 0.5

        # Specific indicators
        specific_patterns = [
            r"\b\d+\b",  # Numbers
            r"\"[^\"]+\"",  # Quoted strings
            r"'[^']+'",  # Single quoted
            r"\b[A-Z][a-z]+\b",  # Proper nouns
            r"\b(specifically|exactly|precisely|particular)\b",
        ]

        for pattern in specific_patterns:
            matches = re.findall(pattern, text)
            specificity += min(0.1, len(matches) * 0.02)

        # Vague indicators
        vague_words = [
            "something",
            "anything",
            "whatever",
            "somehow",
            "maybe",
            "perhaps",
            "kind of",
        ]
        for word in vague_words:
            if word in text.lower():
                specificity -= 0.1

        return max(0.0, min(1.0, specificity))

    def _extract_entities(self, text: str) -> list[tuple[str, str]]:
        """Extract named entities using spaCy."""
        try:
            if self._nlp is None:
                # Use centralized registry for shared model instances
                from headroom.models.ml_models import MLModelRegistry

                self._nlp = MLModelRegistry.get_spacy()

            assert self._nlp is not None
            doc = self._nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        except Exception as e:
            logger.debug(f"NER failed: {e}")
            return []

    def _check_factual_recall(self, text: str) -> bool:
        """Check if prompt requires factual knowledge."""
        factual_patterns = [
            r"\bwhat is\b",
            r"\bwho is\b",
            r"\bwhen did\b",
            r"\bwhere is\b",
            r"\bhow many\b",
            r"\bdefine\b",
            r"\bfact\b",
        ]
        return any(re.search(p, text) for p in factual_patterns)

    def _check_reasoning(self, text: str) -> bool:
        """Check if prompt requires reasoning."""
        reasoning_patterns = [
            r"\bwhy\b",
            r"\bhow\b",
            r"\bexplain\b",
            r"\breason\b",
            r"\banalyze\b",
            r"\bcompare\b",
            r"\bevaluate\b",
        ]
        return any(re.search(p, text) for p in reasoning_patterns)

    def _check_creativity(self, text: str) -> bool:
        """Check if prompt requires creativity."""
        creative_patterns = [
            r"\bcreate\b",
            r"\bimagine\b",
            r"\bwrite a story\b",
            r"\bpoem\b",
            r"\bfiction\b",
            r"\binvent\b",
        ]
        return any(re.search(p, text) for p in creative_patterns)

    def _check_structured_output(self, text: str) -> bool:
        """Check if prompt requests structured output."""
        structured_patterns = [
            r"\bjson\b",
            r"\bxml\b",
            r"\bcsv\b",
            r"\btable\b",
            r"\blist\b",
            r"\bbullet\b",
            r"\bformat\b",
        ]
        return any(re.search(p, text) for p in structured_patterns)

    def _extract_length_request(self, text: str) -> dict[str, Any]:
        """Extract explicit length requests from text."""
        result: dict[str, Any] = {}

        for pattern, length_type in self.LENGTH_PATTERNS:
            match = pattern.search(text)
            if match:
                if length_type in ("brief", "short", "detailed", "comprehensive"):
                    result["type"] = length_type
                else:
                    try:
                        count = int(match.group(1))
                        result[length_type] = count
                        result["type"] = length_type
                    except (ValueError, IndexError):
                        pass

        return result

    def _detect_sentiment(self, words: list[str]) -> str:
        """Detect overall sentiment of prompt."""
        pos_count = sum(1 for w in words if w in self.POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in self.NEGATIVE_WORDS)

        if pos_count > neg_count + 1:
            return "positive"
        elif neg_count > pos_count + 1:
            return "negative"
        return "neutral"

    def _estimate_formality(self, text: str) -> float:
        """Estimate formality level (0 = casual, 1 = formal)."""
        formality = 0.5

        # Formal indicators
        formal_patterns = [
            r"\bplease\b",
            r"\bkindly\b",
            r"\bwould you\b",
            r"\bcould you\b",
            r"\bi would like\b",
            r"\bregards\b",
        ]
        for pattern in formal_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                formality += 0.1

        # Casual indicators
        casual_patterns = [
            r"\bhey\b",
            r"\bhi\b",
            r"\bthanks\b",
            r"\byeah\b",
            r"\bnope\b",
            r"\bcool\b",
            r"!{2,}",
        ]
        for pattern in casual_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                formality -= 0.1

        return max(0.0, min(1.0, formality))

    def _extract_keywords(self, text: str, n: int = 10) -> list[str]:
        """Extract top N keywords using TF-IDF approximation."""
        # Simple keyword extraction (proper implementation would use TF-IDF)
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        word_freq = Counter(words)

        # Filter stop words
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "had",
            "her",
            "was",
            "one",
            "our",
            "out",
            "has",
            "have",
            "been",
            "were",
            "will",
            "with",
            "that",
            "this",
            "from",
            "they",
            "what",
            "which",
            "their",
            "there",
            "about",
        }

        keywords = [word for word, _ in word_freq.most_common(n * 2) if word not in stop_words][:n]

        return keywords

    def _detect_format(self, text: str) -> PromptFormat:
        """Detect the format of the prompt."""
        text_stripped = text.strip()

        # Question
        if text_stripped.endswith("?"):
            return PromptFormat.QUESTION

        # Multi-turn (has role markers)
        if re.search(r"^(User|Human|Assistant|AI):", text, re.MULTILINE | re.IGNORECASE):
            return PromptFormat.MULTI_TURN

        # Template (has placeholders)
        if re.search(r"\{[^}]+\}|\[.*\]|<.*>", text):
            return PromptFormat.TEMPLATE

        # Raw data (mostly code or JSON)
        code_ratio = len(re.findall(r"[{}\[\]();=<>]", text)) / max(1, len(text))
        if code_ratio > 0.1:
            return PromptFormat.RAW_DATA

        # Context + query
        if len(text) > 500 and text_stripped.endswith("?"):
            return PromptFormat.CONTEXT_QUERY

        # Default to instruction
        return PromptFormat.INSTRUCTION


class EmbeddingExtractor(BaseFeatureExtractor):
    """Extracts embedding-based features.

    Requires sentence-transformers for full functionality.
    """

    # Pre-computed cluster centers for common patterns
    # These would be learned from training data in production
    DEFAULT_CLUSTERS: ClassVar[dict[str, list[float]]] = {}

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        cluster_centers: dict[str, list[float]] | None = None,
    ):
        """Initialize embedding extractor.

        Args:
            model_name: Sentence transformer model name. Uses config default if None.
            device: Device for model ('cpu', 'cuda', 'mps', or None for auto).
            cluster_centers: Pre-computed cluster centers for similarity.
        """
        self.model_name = model_name or ML_MODEL_DEFAULTS.sentence_transformer
        self.device = device
        self.cluster_centers = cluster_centers or self.DEFAULT_CLUSTERS
        self._model: SentenceTransformer | None = None

    @staticmethod
    def is_available() -> bool:
        """Check if sentence-transformers is installed."""
        try:
            import sentence_transformers  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_model(self) -> SentenceTransformer:
        """Get or load the sentence transformer model."""
        if self._model is not None:
            return self._model

        if not self.is_available():
            raise RuntimeError(
                "EmbeddingExtractor requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            )

        # Use centralized registry for shared model instances
        from headroom.models.ml_models import MLModelRegistry

        self._model = MLModelRegistry.get_sentence_transformer(self.model_name, self.device)
        return self._model

    def extract(self, text: str, **kwargs: Any) -> EmbeddingFeatures:
        """Extract embedding-based features."""
        features = EmbeddingFeatures()

        if not text or not self.is_available():
            return features

        try:
            model = self._get_model()
            embedding = model.encode(
                text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )

            # Store raw embedding
            features.raw_embedding = embedding.tolist()
            features.embedding_dim = len(embedding)

            # Embedding statistics
            import numpy as np

            features.embedding_norm = float(np.linalg.norm(embedding))
            features.embedding_mean = float(np.mean(embedding))
            features.embedding_std = float(np.std(embedding))
            features.embedding_max = float(np.max(embedding))
            features.embedding_min = float(np.min(embedding))

            # Embedding entropy
            # Normalize to probabilities and compute entropy
            abs_emb = np.abs(embedding)
            probs = abs_emb / (abs_emb.sum() + 1e-10)
            features.embedding_entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

            # Similarity to cluster centers (if available)
            if self.cluster_centers:
                for cluster_name, center in self.cluster_centers.items():
                    center_arr = np.array(center)
                    if len(center_arr) == len(embedding):
                        similarity = float(np.dot(embedding, center_arr))
                        if cluster_name == "short_response":
                            features.similarity_to_short_response_cluster = similarity
                        elif cluster_name == "long_response":
                            features.similarity_to_long_response_cluster = similarity
                        elif cluster_name == "code":
                            features.similarity_to_code_cluster = similarity
                        elif cluster_name == "explanation":
                            features.similarity_to_explanation_cluster = similarity
                        elif cluster_name == "list":
                            features.similarity_to_list_cluster = similarity

        except Exception as e:
            logger.warning(f"Embedding extraction failed: {e}")

        return features


class MetaExtractor(BaseFeatureExtractor):
    """Extracts meta features related to model and context."""

    # Model family patterns
    MODEL_FAMILIES = {
        "gpt": ["gpt-4", "gpt-3.5", "gpt-4o", "o1", "o3"],
        "claude": ["claude-3", "claude-2", "claude-instant"],
        "llama": ["llama-3", "llama-2", "llama"],
        "mistral": ["mistral", "mixtral"],
        "gemini": ["gemini", "palm"],
    }

    # Model context limits (approximate)
    CONTEXT_LIMITS = {
        "gpt-4o": 128000,
        "gpt-4-turbo": 128000,
        "gpt-4": 8192,
        "gpt-3.5-turbo": 16385,
        "claude-3-opus": 200000,
        "claude-3-sonnet": 200000,
        "claude-3-haiku": 200000,
        "llama-3-70b": 8192,
        "mistral-large": 32768,
        "gemini-pro": 32768,
    }

    def __init__(self, tokenizer: Any | None = None):
        """Initialize meta extractor.

        Args:
            tokenizer: Optional tokenizer for exact token counts.
        """
        self.tokenizer = tokenizer

    def extract(
        self,
        text: str,
        model: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        system_prompt: str = "",
        conversation_turn: int = 0,
        cumulative_tokens: int = 0,
        **kwargs: Any,
    ) -> MetaFeatures:
        """Extract meta features.

        Args:
            text: The prompt text.
            model: Model name being used.
            temperature: Generation temperature.
            max_tokens: Max tokens setting.
            top_p: Top-p sampling parameter.
            system_prompt: System prompt if any.
            conversation_turn: Current turn in conversation.
            cumulative_tokens: Total tokens so far.
        """
        features = MetaFeatures()

        # Model information
        features.model_name = model
        features.model_family = self._detect_model_family(model)
        features.model_size_category = self._detect_model_size(model)
        features.model_context_limit = self._get_context_limit(model)

        # Generation settings
        features.temperature = temperature
        features.max_tokens_setting = max_tokens
        features.top_p = top_p

        # Context utilization
        prompt_tokens = len(text) // 4  # Rough estimate
        if self.tokenizer:
            try:
                prompt_tokens = self.tokenizer.count_text(text)
            except Exception:
                pass

        if features.model_context_limit > 0:
            features.prompt_context_ratio = prompt_tokens / features.model_context_limit
            features.available_output_tokens = features.model_context_limit - prompt_tokens
            if max_tokens:
                features.available_output_tokens = min(features.available_output_tokens, max_tokens)

        # Prompt hash
        features.prompt_hash = hashlib.md5(text.encode()).hexdigest()[:16]  # nosec B324
        features.prompt_signature = self._compute_signature(text)

        # Conversation features
        features.is_first_turn = conversation_turn == 0
        features.conversation_turn_number = conversation_turn
        features.cumulative_context_tokens = cumulative_tokens

        # System prompt features
        if system_prompt:
            features.system_prompt_length = len(system_prompt)
            features.system_prompt_token_estimate = len(system_prompt) // 4
            features.has_output_constraints_in_system = self._check_output_constraints(
                system_prompt
            )

        return features

    def _detect_model_family(self, model: str) -> str:
        """Detect model family from name."""
        model_lower = model.lower()
        for family, patterns in self.MODEL_FAMILIES.items():
            if any(p in model_lower for p in patterns):
                return family
        return "unknown"

    def _detect_model_size(self, model: str) -> str:
        """Detect model size category."""
        model_lower = model.lower()

        if any(s in model_lower for s in ["7b", "8b", "small", "mini", "haiku"]):
            return "small"
        elif any(s in model_lower for s in ["13b", "medium", "sonnet"]):
            return "medium"
        elif any(s in model_lower for s in ["70b", "large", "opus"]):
            return "large"
        elif any(s in model_lower for s in ["turbo", "4o"]):
            return "large"

        return "medium"  # Default assumption

    def _get_context_limit(self, model: str) -> int:
        """Get context limit for model."""
        model_lower = model.lower()

        for known_model, limit in self.CONTEXT_LIMITS.items():
            if known_model in model_lower:
                return limit

        # Default limits by family
        family = self._detect_model_family(model)
        family_defaults = {
            "gpt": 8192,
            "claude": 100000,
            "llama": 8192,
            "mistral": 32768,
            "gemini": 32768,
        }
        return family_defaults.get(family, 8192)

    def _compute_signature(self, text: str) -> str:
        """Compute a structural signature of the prompt."""
        # Simple signature based on structure
        features = []

        if "?" in text:
            features.append("Q")
        if re.search(r"^\d+\.", text, re.MULTILINE):
            features.append("L")
        if "```" in text:
            features.append("C")
        if len(text) > 1000:
            features.append("X")

        return "".join(features) or "B"  # B = basic

    def _check_output_constraints(self, text: str) -> bool:
        """Check if system prompt has output constraints."""
        constraint_patterns = [
            r"\bmax\s*\d+\s*words?\b",
            r"\bkeep.*short\b",
            r"\bbrief\b",
            r"\bconcise\b",
            r"\bno more than\b",
            r"\blimit\s+to\b",
        ]
        return any(re.search(p, text, re.IGNORECASE) for p in constraint_patterns)


# =============================================================================
# MAIN FEATURE EXTRACTOR
# =============================================================================


class PromptFeatureExtractor:
    """Complete feature extractor for LLM output length prediction.

    This class orchestrates all feature extractors and provides a unified
    interface for extracting features from prompts.

    Example:
        extractor = PromptFeatureExtractor()

        # Basic extraction
        features = extractor.extract("What is machine learning?")

        # With model context
        features = extractor.extract(
            prompt="Explain quantum computing",
            model="gpt-4o",
            temperature=0.7,
            system_prompt="You are a helpful assistant."
        )

        # Get feature vector for ML
        vector = features.to_vector()
        names = PromptFeatures.feature_names()
    """

    def __init__(
        self,
        tokenizer: Any | None = None,
        use_embeddings: bool = True,
        use_ner: bool = False,
        embedding_model: str | None = None,
        cluster_centers: dict[str, list[float]] | None = None,
    ):
        """Initialize the feature extractor.

        Args:
            tokenizer: Optional tokenizer for exact token counts.
                Should have count_text(str) -> int method.
            use_embeddings: Whether to extract embedding features.
                Requires sentence-transformers.
            use_ner: Whether to use NER for entity extraction.
                Requires spaCy.
            embedding_model: Sentence transformer model name. Uses config default if None.
            cluster_centers: Pre-computed cluster centers for similarity.
        """
        self.text_extractor = TextStatisticsExtractor(tokenizer=tokenizer)
        self.structural_extractor = StructuralExtractor()
        self.semantic_extractor = SemanticExtractor(use_ner=use_ner)
        self.meta_extractor = MetaExtractor(tokenizer=tokenizer)

        self.use_embeddings = use_embeddings
        self.embedding_extractor: EmbeddingExtractor | None
        if use_embeddings:
            self.embedding_extractor = EmbeddingExtractor(
                model_name=embedding_model, cluster_centers=cluster_centers
            )
        else:
            self.embedding_extractor = None

        # Cache for repeated extractions
        self._cache: dict[str, PromptFeatures] = {}
        self._cache_max_size = 1000

    def extract(
        self,
        prompt: str,
        model: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        top_p: float | None = None,
        system_prompt: str = "",
        conversation_turn: int = 0,
        cumulative_tokens: int = 0,
        use_cache: bool = True,
    ) -> PromptFeatures:
        """Extract all features from a prompt.

        Args:
            prompt: The prompt text to analyze.
            model: Model name (for meta features).
            temperature: Generation temperature setting.
            max_tokens: Max tokens setting.
            top_p: Top-p sampling parameter.
            system_prompt: System prompt if any.
            conversation_turn: Current turn number (0 = first).
            cumulative_tokens: Total tokens in conversation so far.
            use_cache: Whether to use caching.

        Returns:
            PromptFeatures containing all extracted features.
        """
        # Check cache
        cache_key = hashlib.md5(f"{prompt}:{model}:{system_prompt}".encode()).hexdigest()  # nosec B324

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Extract all feature categories
        features = PromptFeatures(
            original_prompt=prompt,
            extraction_timestamp=str(__import__("datetime").datetime.now()),
        )

        # 1. Text statistics
        features.text_statistics = self.text_extractor.extract(prompt)

        # 2. Structural features
        features.structural = self.structural_extractor.extract(prompt)

        # 3. Semantic features
        features.semantic = self.semantic_extractor.extract(prompt)

        # 4. Embedding features (optional)
        if self.embedding_extractor and self.use_embeddings:
            features.embedding = self.embedding_extractor.extract(prompt)

        # 5. Meta features
        features.meta = self.meta_extractor.extract(
            text=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            system_prompt=system_prompt,
            conversation_turn=conversation_turn,
            cumulative_tokens=cumulative_tokens,
        )

        # Cache result
        if use_cache:
            if len(self._cache) >= self._cache_max_size:
                # Simple cache eviction: clear half
                keys_to_remove = list(self._cache.keys())[: self._cache_max_size // 2]
                for k in keys_to_remove:
                    del self._cache[k]
            self._cache[cache_key] = features

        return features

    def extract_batch(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[PromptFeatures]:
        """Extract features for multiple prompts.

        More efficient than calling extract() in a loop when using embeddings.

        Args:
            prompts: List of prompts to analyze.
            **kwargs: Additional arguments passed to extract().

        Returns:
            List of PromptFeatures, one per prompt.
        """
        results = []

        # For embeddings, batch encode if possible
        if self.embedding_extractor and self.use_embeddings:
            try:
                model = self.embedding_extractor._get_model()
                embeddings = model.encode(
                    prompts,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )

                for i, prompt in enumerate(prompts):
                    features = self.extract(prompt, use_cache=False, **kwargs)
                    # Override with batch-computed embedding
                    features.embedding.raw_embedding = embeddings[i].tolist()
                    results.append(features)

                return results

            except Exception as e:
                logger.warning(f"Batch embedding failed, falling back: {e}")

        # Fallback: sequential extraction
        for prompt in prompts:
            results.append(self.extract(prompt, **kwargs))

        return results

    def get_feature_names(
        self, include_raw_embedding: bool = False, embedding_dim: int = 384
    ) -> list[str]:
        """Get ordered list of feature names.

        Args:
            include_raw_embedding: Whether to include raw embedding dimensions.
            embedding_dim: Dimension of embeddings (for naming).

        Returns:
            List of feature names matching to_vector() output.
        """
        return PromptFeatures.feature_names(
            include_raw_embedding=include_raw_embedding, embedding_dim=embedding_dim
        )

    def clear_cache(self) -> None:
        """Clear the feature cache."""
        self._cache.clear()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def extract_features(
    prompt: str,
    model: str = "",
    **kwargs: Any,
) -> PromptFeatures:
    """Convenience function for one-off feature extraction.

    Args:
        prompt: The prompt to analyze.
        model: Model name for meta features.
        **kwargs: Additional arguments for PromptFeatureExtractor.extract().

    Returns:
        PromptFeatures object.
    """
    extractor = PromptFeatureExtractor(use_embeddings=False)
    return extractor.extract(prompt, model=model, **kwargs)


def get_feature_vector(
    prompt: str,
    include_raw_embedding: bool = False,
    **kwargs: Any,
) -> list[float]:
    """Get feature vector directly.

    Args:
        prompt: The prompt to analyze.
        include_raw_embedding: Whether to include raw embedding.
        **kwargs: Additional arguments.

    Returns:
        Feature vector as list of floats.
    """
    features = extract_features(prompt, **kwargs)
    return features.to_vector(include_raw_embedding=include_raw_embedding)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Demo usage
    extractor = PromptFeatureExtractor(use_embeddings=False)

    test_prompts = [
        "What is machine learning?",
        "Write a detailed essay about the history of artificial intelligence, "
        "including its origins, key milestones, and future predictions. "
        "Please include at least 5 paragraphs.",
        "Fix this code:\n```python\ndef hello():\n    print('world)\n```",
        "1. Compare Python and JavaScript\n2. List pros and cons\n3. Give examples",
    ]

    for prompt in test_prompts:
        print(f"\n{'=' * 60}")
        print(f"Prompt: {prompt[:50]}...")
        print("=" * 60)

        features = extractor.extract(prompt, model="gpt-4o")

        print("\nText Statistics:")
        print(f"  - Words: {features.text_statistics.word_count}")
        print(f"  - Tokens (est): {features.text_statistics.token_count_estimate}")
        print(f"  - Vocabulary richness: {features.text_statistics.vocabulary_richness:.2f}")
        print(f"  - Compression ratio: {features.text_statistics.compression_ratio:.2f}")

        print("\nStructural:")
        print(f"  - Is question: {features.structural.is_question}")
        print(f"  - Code blocks: {features.structural.code_block_count}")
        print(f"  - List items: {features.structural.total_list_items}")

        print("\nSemantic:")
        print(f"  - Task type: {features.semantic.primary_task_type.value}")
        print(f"  - Domain: {features.semantic.primary_domain.value}")
        print(f"  - Complexity: {features.semantic.complexity_level.value}")

        print("\nMeta:")
        print(f"  - Prompt hash: {features.meta.prompt_hash}")
        print(f"  - Context ratio: {features.meta.prompt_context_ratio:.4f}")

        vector = features.to_vector()
        print(f"\nFeature vector length: {len(vector)}")
