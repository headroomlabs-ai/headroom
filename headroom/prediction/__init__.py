"""LLM Output Length Prediction Module.

This module provides comprehensive feature extraction and prediction
capabilities for estimating LLM response lengths from input prompts.

Features are organized into 5 categories:
1. Text Statistics - Length, vocabulary, compression metrics
2. Structural - Questions, lists, code blocks, formatting
3. Semantic - Task type, domain, complexity indicators
4. Embedding - Neural embeddings and similarity scores
5. Meta - Model settings, historical patterns

Example:
    from headroom.prediction import PromptFeatureExtractor, extract_features

    # Full extractor (with embeddings)
    extractor = PromptFeatureExtractor(use_embeddings=True)
    features = extractor.extract("What is machine learning?", model="gpt-4o")

    # Quick extraction (no embeddings)
    features = extract_features("Explain quantum computing")

    # Get ML-ready vector
    vector = features.to_vector()
    names = features.feature_names()

Install full dependencies:
    pip install headroom[prediction]

This installs:
    - sentence-transformers (for embedding features)
    - spacy (for NER, optional)
"""

from .feature_extractor import (
    ComplexityLevel,
    DomainType,
    EmbeddingExtractor,
    EmbeddingFeatures,
    MetaExtractor,
    MetaFeatures,
    # Main extractor
    PromptFeatureExtractor,
    # Feature dataclasses
    PromptFeatures,
    PromptFormat,
    SemanticExtractor,
    SemanticFeatures,
    StructuralExtractor,
    StructuralFeatures,
    # Enums
    TaskType,
    # Individual extractors
    TextStatisticsExtractor,
    TextStatisticsFeatures,
    # Utility functions
    extract_features,
    get_feature_vector,
)

__all__ = [
    # Main extractor
    "PromptFeatureExtractor",
    # Individual extractors
    "TextStatisticsExtractor",
    "StructuralExtractor",
    "SemanticExtractor",
    "EmbeddingExtractor",
    "MetaExtractor",
    # Feature dataclasses
    "PromptFeatures",
    "TextStatisticsFeatures",
    "StructuralFeatures",
    "SemanticFeatures",
    "EmbeddingFeatures",
    "MetaFeatures",
    # Enums
    "TaskType",
    "DomainType",
    "ComplexityLevel",
    "PromptFormat",
    # Utility functions
    "extract_features",
    "get_feature_vector",
]
