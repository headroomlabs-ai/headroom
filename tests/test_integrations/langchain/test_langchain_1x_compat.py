"""Regression tests for the LangChain 1.x compatibility fixes.

Each test here corresponds to a defect that shipped in 0.37.0 and was found by
running the documented examples against langchain-core 1.6:

2. ``wrap_tools_with_headroom`` produced tools that raised ``TypeError`` on
   invoke, because ``StructuredTool`` calls its ``func`` with unpacked kwargs
   while ``BaseTool.invoke`` takes a single input.
3. The wrapped tool lost the original ``args_schema``, so a model saw a tool
   with no parameters.
4. ``HeadroomDocumentCompressor`` silently subclassed a local stub rather than
   LangChain's ``BaseDocumentCompressor``, so retrievers rejected it.
"""

import asyncio
import json

import pytest

try:
    from langchain_core.tools import tool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

pytestmark = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")


BIG_RESULT = json.dumps(
    {
        "results": [
            {"id": i, "user": f"user{i}", "plan": "pro", "status": "active"} for i in range(300)
        ],
        "total": 300,
    }
)


@tool
def query_database(query: str) -> str:
    """Query the users database. Returns JSON rows."""
    return BIG_RESULT


class TestWrappedToolIsUsable:
    """Defects 2 and 3: the wrapped tool must invoke, and keep its schema."""

    def test_invoke_with_keyword_arguments(self):
        from headroom.integrations import wrap_tools_with_headroom

        wrapped = wrap_tools_with_headroom([query_database], min_chars_to_compress=1000)[0]
        out = wrapped.invoke({"query": "signups"})

        assert isinstance(out, str) and out
        assert len(out) < len(BIG_RESULT)

    def test_argument_schema_is_preserved(self):
        from headroom.integrations import wrap_tools_with_headroom

        wrapped = wrap_tools_with_headroom([query_database])[0]

        assert sorted(wrapped.args_schema.model_fields) == ["query"], (
            "the wrapped tool advertises different parameters than the original, "
            "so a model cannot call it correctly"
        )

    def test_async_invoke_compresses(self):
        from headroom.integrations import wrap_tools_with_headroom

        wrapped = wrap_tools_with_headroom([query_database], min_chars_to_compress=1000)[0]
        out = asyncio.run(wrapped.ainvoke({"query": "signups"}))

        assert len(out) < len(BIG_RESULT)

    def test_unusable_args_schema_falls_back_to_inference(self):
        """A tool carrying a schema LangChain cannot use must still wrap."""
        from headroom.integrations.langchain.agents import HeadroomToolWrapper

        class OddTool:
            name = "odd"
            description = "a tool with a schema LangChain will not accept"
            args_schema = object()

            def invoke(self, value):
                return "small"

        wrapper = HeadroomToolWrapper(tool=OddTool())
        assert wrapper.as_langchain_tool().name == "odd"


class TestDocumentCompressorBaseClass:
    """Defect 4: the compressor must be a real LangChain compressor."""

    def test_is_langchain_base_document_compressor(self):
        from langchain_core.documents.compressor import BaseDocumentCompressor

        from headroom.integrations import HeadroomDocumentCompressor

        compressor = HeadroomDocumentCompressor(max_documents=10)

        assert isinstance(compressor, BaseDocumentCompressor), (
            "HeadroomDocumentCompressor fell back to the local stub base class; "
            "ContextualCompressionRetriever validates against the real one and "
            "would reject this compressor"
        )

    def test_compresses_down_to_max_documents(self):
        from langchain_core.documents import Document

        from headroom.integrations import HeadroomDocumentCompressor

        docs = [Document(page_content=f"Python is a language. item {i}") for i in range(50)]
        out = HeadroomDocumentCompressor(max_documents=10, min_relevance=0.0).compress_documents(
            docs, "What is Python?"
        )

        assert len(out) <= 10
