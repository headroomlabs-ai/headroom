"""Regression tests for the LangChain 1.x compatibility fixes.

Each test here corresponds to a defect that shipped in 0.37.0 and was found by
running the documented examples against langchain-core 1.6:

1. ``HeadroomChatModel.bind_tools()`` returned a model that emitted no tool
   calls, because ``_generate`` called the private method on the
   ``RunnableBinding`` and the bound kwargs were dropped.
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
from typing import Any

import pytest

try:
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage
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


class _RecordingModel(GenericFakeChatModel):
    """Fake model that records the kwargs its _generate actually received."""

    last_kwargs: dict[str, Any] = {}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        type(self).last_kwargs = dict(kwargs)
        kwargs.pop("tools", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        # Mirrors what real providers do: return a RunnableBinding carrying the
        # tools in .kwargs rather than a new model instance.
        return self.bind(tools=list(tools), **kwargs)


class TestBindToolsSurvivesWrapping:
    """Defect 1: bound kwargs must reach the underlying model."""

    def test_bound_tools_reach_generate(self):
        from headroom.integrations import HeadroomChatModel

        _RecordingModel.last_kwargs = {}
        inner = _RecordingModel(messages=iter([AIMessage("ok")]))
        wrapped = HeadroomChatModel(inner).bind_tools([query_database])

        wrapped.invoke("call the tool")

        assert "tools" in _RecordingModel.last_kwargs, (
            "bind_tools kwargs were dropped before reaching the wrapped model; "
            "an agent built on this model would never call a tool"
        )

    def test_unbound_model_passes_no_tools(self):
        from headroom.integrations import HeadroomChatModel

        _RecordingModel.last_kwargs = {}
        inner = _RecordingModel(messages=iter([AIMessage("ok")]))
        HeadroomChatModel(inner).invoke("hello")

        assert "tools" not in _RecordingModel.last_kwargs

    def test_unwrap_binding_ignores_non_bindings(self):
        from headroom.integrations import HeadroomChatModel

        inner = GenericFakeChatModel(messages=iter([AIMessage("ok")]))
        model, kwargs = HeadroomChatModel._unwrap_binding(inner)

        assert model is inner
        assert kwargs == {}


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
