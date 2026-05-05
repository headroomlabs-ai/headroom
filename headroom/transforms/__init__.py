"""Transform modules for Headroom SDK."""

from __future__ import annotations

import importlib.util
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Expose concrete types to static analysis while keeping runtime imports lazy.
    from headroom.transforms.anchor_selector import (  # noqa: F401
        AnchorSelector,
        AnchorStrategy,
        AnchorWeights,
        DataPattern,
        calculate_information_score,
        compute_item_hash,
    )
    from headroom.transforms.base import Transform  # noqa: F401
    from headroom.transforms.cache_aligner import CacheAligner  # noqa: F401
    from headroom.transforms.code_compressor import (  # noqa: F401
        CodeAwareCompressor,
        CodeCompressionResult,
        CodeCompressorConfig,
        CodeLanguage,
        DocstringMode,
        detect_language,
        is_tree_sitter_available,
    )
    from headroom.transforms.content_detector import (  # noqa: F401
        ContentType,
        DetectionResult,
        detect_content_type,
    )
    from headroom.transforms.content_router import (  # noqa: F401
        CompressionStrategy,
        ContentRouter,
        ContentRouterConfig,
        RouterCompressionResult,
    )
    from headroom.transforms.diff_compressor import (  # noqa: F401
        DiffCompressionResult,
        DiffCompressor,
        DiffCompressorConfig,
    )
    from headroom.transforms.html_extractor import (  # noqa: F401
        HTMLExtractionResult,
        HTMLExtractor,
        HTMLExtractorConfig,
        is_html_content,
    )
    from headroom.transforms.log_compressor import (  # noqa: F401
        LogCompressionResult,
        LogCompressor,
        LogCompressorConfig,
    )
    from headroom.transforms.pipeline import TransformPipeline  # noqa: F401
    from headroom.transforms.search_compressor import (  # noqa: F401
        SearchCompressionResult,
        SearchCompressor,
        SearchCompressorConfig,
    )
    from headroom.transforms.smart_crusher import SmartCrusher, SmartCrusherConfig  # noqa: F401

_HTML_EXTRACTOR_AVAILABLE = importlib.util.find_spec("trafilatura") is not None

__all__ = [
    # Base
    "Transform",
    "TransformPipeline",
    # Anchor selection
    "AnchorSelector",
    "AnchorStrategy",
    "AnchorWeights",
    "DataPattern",
    "calculate_information_score",
    "compute_item_hash",
    # JSON compression
    "SmartCrusher",
    "SmartCrusherConfig",
    # Text compression (coding tasks)
    "ContentType",
    "DetectionResult",
    "detect_content_type",
    "SearchCompressor",
    "SearchCompressorConfig",
    "SearchCompressionResult",
    "LogCompressor",
    "LogCompressorConfig",
    "LogCompressionResult",
    "DiffCompressor",
    "DiffCompressorConfig",
    "DiffCompressionResult",
    # Code-aware compression (AST-based)
    "CodeAwareCompressor",
    "CodeCompressorConfig",
    "CodeCompressionResult",
    "CodeLanguage",
    "DocstringMode",
    "detect_language",
    "is_tree_sitter_available",
    # Content routing
    "ContentRouter",
    "ContentRouterConfig",
    "RouterCompressionResult",
    "CompressionStrategy",
    # Other transforms
    "CacheAligner",
    # HTML extraction (optional)
    "_HTML_EXTRACTOR_AVAILABLE",
]

# Conditionally add HTML extractor exports
if _HTML_EXTRACTOR_AVAILABLE:
    __all__.extend(
        [
            "HTMLExtractor",
            "HTMLExtractorConfig",
            "HTMLExtractionResult",
            "is_html_content",
        ]
    )

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    # Base
    "Transform": ("headroom.transforms.base", "Transform"),
    "TransformPipeline": ("headroom.transforms.pipeline", "TransformPipeline"),
    # Anchor selection
    "AnchorSelector": ("headroom.transforms.anchor_selector", "AnchorSelector"),
    "AnchorStrategy": ("headroom.transforms.anchor_selector", "AnchorStrategy"),
    "AnchorWeights": ("headroom.transforms.anchor_selector", "AnchorWeights"),
    "DataPattern": ("headroom.transforms.anchor_selector", "DataPattern"),
    "calculate_information_score": (
        "headroom.transforms.anchor_selector",
        "calculate_information_score",
    ),
    "compute_item_hash": ("headroom.transforms.anchor_selector", "compute_item_hash"),
    # JSON compression
    "SmartCrusher": ("headroom.transforms.smart_crusher", "SmartCrusher"),
    "SmartCrusherConfig": ("headroom.transforms.smart_crusher", "SmartCrusherConfig"),
    # Text compression (coding tasks)
    "ContentType": ("headroom.transforms.content_detector", "ContentType"),
    "DetectionResult": ("headroom.transforms.content_detector", "DetectionResult"),
    "detect_content_type": ("headroom.transforms.content_detector", "detect_content_type"),
    "SearchCompressor": ("headroom.transforms.search_compressor", "SearchCompressor"),
    "SearchCompressorConfig": (
        "headroom.transforms.search_compressor",
        "SearchCompressorConfig",
    ),
    "SearchCompressionResult": (
        "headroom.transforms.search_compressor",
        "SearchCompressionResult",
    ),
    "LogCompressor": ("headroom.transforms.log_compressor", "LogCompressor"),
    "LogCompressorConfig": ("headroom.transforms.log_compressor", "LogCompressorConfig"),
    "LogCompressionResult": ("headroom.transforms.log_compressor", "LogCompressionResult"),
    "DiffCompressor": ("headroom.transforms.diff_compressor", "DiffCompressor"),
    "DiffCompressorConfig": ("headroom.transforms.diff_compressor", "DiffCompressorConfig"),
    "DiffCompressionResult": (
        "headroom.transforms.diff_compressor",
        "DiffCompressionResult",
    ),
    # Code-aware compression (AST-based)
    "CodeAwareCompressor": ("headroom.transforms.code_compressor", "CodeAwareCompressor"),
    "CodeCompressorConfig": ("headroom.transforms.code_compressor", "CodeCompressorConfig"),
    "CodeCompressionResult": (
        "headroom.transforms.code_compressor",
        "CodeCompressionResult",
    ),
    "CodeLanguage": ("headroom.transforms.code_compressor", "CodeLanguage"),
    "DocstringMode": ("headroom.transforms.code_compressor", "DocstringMode"),
    "detect_language": ("headroom.transforms.code_compressor", "detect_language"),
    "is_tree_sitter_available": (
        "headroom.transforms.code_compressor",
        "is_tree_sitter_available",
    ),
    # Content routing
    "ContentRouter": ("headroom.transforms.content_router", "ContentRouter"),
    "ContentRouterConfig": ("headroom.transforms.content_router", "ContentRouterConfig"),
    "RouterCompressionResult": (
        "headroom.transforms.content_router",
        "RouterCompressionResult",
    ),
    "CompressionStrategy": ("headroom.transforms.content_router", "CompressionStrategy"),
    # Other transforms
    "CacheAligner": ("headroom.transforms.cache_aligner", "CacheAligner"),
    # HTML extraction (optional dependency - requires trafilatura)
    "HTMLExtractor": ("headroom.transforms.html_extractor", "HTMLExtractor"),
    "HTMLExtractorConfig": ("headroom.transforms.html_extractor", "HTMLExtractorConfig"),
    "HTMLExtractionResult": ("headroom.transforms.html_extractor", "HTMLExtractionResult"),
    "is_html_content": ("headroom.transforms.html_extractor", "is_html_content"),
}


def __getattr__(name: str) -> object:
    if name == "__path__":
        raise AttributeError(name)
    if name == "_HTML_EXTRACTOR_AVAILABLE":
        return _HTML_EXTRACTOR_AVAILABLE

    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
