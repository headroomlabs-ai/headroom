"""Live LangChain integration tests — no mocks, real API keys from .env.

Run with:
  pytest tests/test_integrations/langchain/test_langchain_live.py -v -s
  # Or with env loaded:
  set -a && source .env && set +a && pytest tests/test_integrations/langchain/test_langchain_live.py -v -s

Requires: OPENAI_API_KEY and/or ANTHROPIC_API_KEY in environment (e.g. from .env).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Load .env from project root if present
_project_root = Path(__file__).resolve().parents[3]
_env = _project_root / ".env"
if _env.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_env)
    except ImportError:
        pass

try:
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import tool

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
HAS_OPENAI = bool(OPENAI_KEY)
HAS_ANTHROPIC = bool(ANTHROPIC_KEY)
HAS_ANY_KEY = HAS_OPENAI or HAS_ANTHROPIC

pytestmark = [
    pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed"),
    pytest.mark.skipif(
        not HAS_ANY_KEY, reason="No OPENAI_API_KEY or ANTHROPIC_API_KEY in env (e.g. .env)"
    ),
]


@pytest.fixture
def openai_llm():
    """Real ChatOpenAI if OPENAI_API_KEY is set."""
    if not HAS_OPENAI:
        pytest.skip("OPENAI_API_KEY not set")
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


@pytest.fixture
def anthropic_llm():
    """Real ChatAnthropic if ANTHROPIC_API_KEY is set."""
    if not HAS_ANTHROPIC:
        pytest.skip("ANTHROPIC_API_KEY not set")
    from langchain_anthropic import ChatAnthropic

    # Allow override via env (e.g. claude-sonnet-4-20250514); default to a common current model
    model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
    return ChatAnthropic(model=model, temperature=0)


# --- HeadroomChatModel: invoke (sync) ---


class TestHeadroomChatModelLiveOpenAI:
    """Live tests: HeadroomChatModel wrapping ChatOpenAI."""

    def test_wrap_openai_and_invoke(self, openai_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        messages = [HumanMessage(content="Reply with exactly: OK")]
        response = model.invoke(messages)

        assert response is not None
        assert hasattr(response, "content")
        assert response.content is not None
        assert len(response.content) > 0
        assert len(model._metrics_history) >= 1
        m = model._metrics_history[-1]
        assert m.tokens_before >= 0
        assert m.tokens_after >= 0

    def test_invoke_with_string_input(self, openai_llm):
        """LangChain allows invoke(str); BaseChatModel converts to messages."""
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        response = model.invoke("Say hello in one word.")
        assert response is not None
        assert hasattr(response, "content")
        assert len(response.content) > 0

    def test_system_and_user_messages(self, openai_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        messages = [
            SystemMessage(content="You are a helpful assistant. Be very brief."),
            HumanMessage(content="What is 2+2? One number only."),
        ]
        response = model.invoke(messages)
        assert response.content is not None
        assert "4" in response.content or "four" in response.content.lower()

    def test_get_savings_summary_after_calls(self, openai_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        model.invoke([HumanMessage(content="Hi")])
        summary = model.get_savings_summary()
        assert summary["total_requests"] >= 1
        assert "total_tokens_saved" in summary
        assert "average_savings_percent" in summary


class TestHeadroomChatModelLiveAnthropic:
    """Live tests: HeadroomChatModel wrapping ChatAnthropic.

    If your Anthropic account does not have access to the default model,
    set ANTHROPIC_MODEL=your-model (e.g. claude-3-5-sonnet-20241022) in .env.
    """

    def test_wrap_anthropic_and_invoke(self, anthropic_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(anthropic_llm)
        messages = [HumanMessage(content="Reply with exactly: OK")]
        try:
            response = model.invoke(messages)
        except Exception as e:
            if "404" in str(e) or "not_found" in str(e).lower():
                pytest.skip(f"Anthropic model not available: {e}")
            raise
        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert len(model._metrics_history) >= 1

    def test_provider_detection_anthropic(self, anthropic_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(anthropic_llm)
        _ = model.pipeline
        assert model._provider is not None
        assert "anthropic" in model._provider.__class__.__name__.lower() or "anthropic" in str(
            type(model._provider)
        )


# --- Streaming ---


class TestHeadroomChatModelStreamingLive:
    """Live streaming tests."""

    def test_stream_openai(self, openai_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        messages = [HumanMessage(content="Count from 1 to 3, one number per line.")]
        chunks = list(model.stream(messages))
        assert len(chunks) >= 1
        full = "".join(c.content for c in chunks if c.content)
        assert "1" in full or "2" in full or "3" in full

    @pytest.mark.asyncio
    async def test_astream_openai(self, openai_llm):
        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        messages = [HumanMessage(content="Say 'stream' and nothing else.")]
        count = 0
        async for chunk in model.astream(messages):
            if chunk.content:
                count += 1
        assert count >= 1


# --- Tool calling (real round-trip) ---


class TestHeadroomChatModelToolCallsLive:
    """Live tool-calling tests: bind_tools + invoke with tool use."""

    def test_bind_tools_and_invoke_with_tool_output(self, openai_llm):
        """Simulate agent turn: user -> model (tool call) -> tool result -> model. We compress tool result."""
        from headroom.integrations import HeadroomChatModel

        @tool
        def big_search(query: str) -> str:
            """Search (returns large JSON)."""
            import json

            return json.dumps(
                {
                    "results": [
                        {"id": i, "title": f"Result {i}", "snippet": "x" * 200} for i in range(50)
                    ],
                    "total": 50,
                }
            )

        base = openai_llm.bind_tools([big_search])
        model = HeadroomChatModel(base)

        # User asks something that may trigger tool use
        messages = [
            HumanMessage(
                content="Search for 'python tutorials' and tell me how many results you got."
            ),
        ]
        response = model.invoke(messages)

        assert response is not None
        # Either direct answer or tool_calls
        if response.tool_calls:
            assert len(response.tool_calls) >= 1
            tc = response.tool_calls[0]
            assert "name" in tc or hasattr(tc, "get")
        assert len(model._metrics_history) >= 1

    def test_messages_with_tool_result_compressed(self, openai_llm):
        """Conversation with tool call + large tool result; Headroom should compress the tool result."""
        import json

        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        # Simulate: user -> assistant (tool call) -> tool (large result) -> user (follow-up)
        large_result = json.dumps([{"id": i, "data": "x" * 100} for i in range(100)])
        messages = [
            HumanMessage(content="Get items 1 to 100."),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_items",
                        "args": {"limit": 100},
                        "type": "tool_call",
                    }
                ],
            ),
            ToolMessage(content=large_result, tool_call_id="call_1"),
            HumanMessage(content="How many items did you get? One number only."),
        ]
        response = model.invoke(messages)

        assert response is not None
        assert response.content is not None
        # Optimization should have run (tool content was large)
        assert len(model._metrics_history) >= 1
        last = model._metrics_history[-1]
        assert last.tokens_before >= last.tokens_after or last.tokens_before == last.tokens_after


# --- LCEL chain ---


class TestHeadroomLCELive:
    """Live LCEL chain tests."""

    def test_prompt_pipe_headroom_pipe_llm(self, openai_llm):
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        from headroom.integrations import HeadroomChatModel

        model = HeadroomChatModel(openai_llm)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are helpful. Reply in one short sentence."),
                ("human", "{input}"),
            ]
        )
        chain = prompt | model | StrOutputParser()
        result = chain.invoke({"input": "What is the capital of France?"})
        assert result is not None
        assert "Paris" in result or "paris" in result.lower()


# --- optimize_messages standalone (no LLM call) ---


class TestOptimizeMessagesLive:
    """Live optimize_messages with real Headroom pipeline (no API key needed for this)."""

    def test_optimize_messages_large_conversation(self):
        from headroom.integrations import optimize_messages

        messages = [SystemMessage(content="You are helpful.")]
        for i in range(30):
            messages.append(HumanMessage(content=f"Question {i}: What is {i}?"))
            messages.append(AIMessage(content=f"Answer: {i}."))
        messages.append(HumanMessage(content="Summarize the last answer."))

        optimized, metrics = optimize_messages(messages)
        assert len(optimized) >= 1
        assert metrics["tokens_before"] >= metrics["tokens_after"]
        assert "transforms_applied" in metrics
