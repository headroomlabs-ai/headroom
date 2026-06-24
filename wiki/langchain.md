# LangChain Integration

Headroom provides seamless integration with LangChain, enabling automatic context optimization across all LangChain patterns: chat models, memory, retrievers, agents, and observability.

## Installation

```bash
pip install "headroom-ai[langchain]"
```

This installs Headroom with LangChain dependencies (`langchain-core`).

## Quick Start

### Wrap Any Chat Model (1 Line)

```python
from langchain_openai import ChatOpenAI
from headroom.integrations import HeadroomChatModel

# Wrap your model - that's it!
llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))

# Use exactly like before
response = llm.invoke("Hello!")
```

Headroom automatically:
- Detects the provider (OpenAI, Anthropic, Google)
- Compresses tool outputs in conversation history
- Optimizes for provider caching
- Tracks token savings

### Check Your Savings

```python
# After some usage
print(llm.get_metrics())
# {'tokens_saved': 12500, 'savings_percent': 45.2, 'requests': 50}
```

---

## Integration Patterns

### 1. Chat Model Wrapper

The `HeadroomChatModel` wraps any LangChain `BaseChatModel`:

```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from headroom.integrations import HeadroomChatModel

# OpenAI
llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))

# Anthropic (auto-detected)
llm = HeadroomChatModel(ChatAnthropic(model="claude-3-5-sonnet-20241022"))

# Custom configuration
from headroom import HeadroomConfig, HeadroomMode

config = HeadroomConfig(
    default_mode=HeadroomMode.OPTIMIZE,
    smart_crusher_target_ratio=0.3,  # Target 70% compression
)
llm = HeadroomChatModel(
    ChatOpenAI(model="gpt-4o"),
    headroom_config=config,
)
```

#### Async Support

Full async support for `ainvoke` and `astream`:

```python
# Async invoke
response = await llm.ainvoke("Hello!")

# Async streaming
async for chunk in llm.astream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

#### Tool Calling

Works seamlessly with LangChain tool calling:

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web."""
    return {"results": [...]}  # Large JSON response

llm_with_tools = llm.bind_tools([search])
response = llm_with_tools.invoke("Search for Python tutorials")
# Tool outputs are automatically compressed in subsequent turns
```

---

### 2. Memory Integration

`HeadroomChatMessageHistory` wraps any chat history with automatic compression:

```python
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from headroom.integrations import HeadroomChatMessageHistory

# Wrap any history
base_history = ChatMessageHistory()
compressed_history = HeadroomChatMessageHistory(
    base_history,
    compress_threshold_tokens=4000,  # Compress when over 4K tokens
    keep_recent_turns=5,             # Always keep last 5 turns
)

# Use with any memory class
memory = ConversationBufferMemory(chat_memory=compressed_history)

# Zero changes to your chain!
chain = ConversationChain(llm=llm, memory=memory)
```

**Why this matters**: Long conversations can blow up to 50K+ tokens. HeadroomChatMessageHistory automatically compresses older turns while preserving recent context.

```python
# Check compression stats
print(compressed_history.get_compression_stats())
# {'compression_count': 12, 'total_tokens_saved': 28000}
```

---

### 3. Retriever Integration

`HeadroomDocumentCompressor` filters retrieved documents by relevance:

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.vectorstores import FAISS
from headroom.integrations import HeadroomDocumentCompressor

# Create vector store retriever (retrieve many for recall)
vectorstore = FAISS.from_documents(documents, embeddings)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

# Wrap with Headroom compression (keep best for precision)
compressor = HeadroomDocumentCompressor(
    max_documents=10,      # Keep top 10
    min_relevance=0.3,     # Minimum relevance score
    prefer_diverse=True,   # MMR-style diversity
)

retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# Retrieves 50 docs, returns best 10
docs = retriever.invoke("What is Python?")
```

**Why this matters**: Vector search often returns many marginally-relevant documents. HeadroomDocumentCompressor uses BM25-style scoring to keep only the most relevant ones, reducing context size while improving answer quality.

---

### 4. Agent Tool Wrapping

`wrap_tools_with_headroom` compresses tool outputs for agents:

```python
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import tool
from headroom.integrations import wrap_tools_with_headroom

@tool
def search_database(query: str) -> str:
    """Search the database."""
    # Returns 1000 results as JSON
    return json.dumps({"results": [...], "total": 1000})

@tool
def fetch_logs(service: str) -> str:
    """Fetch service logs."""
    # Returns 500 log entries
    return json.dumps({"logs": [...]})

# Wrap tools with compression
tools = [search_database, fetch_logs]
wrapped_tools = wrap_tools_with_headroom(
    tools,
    min_chars_to_compress=1000,  # Only compress large outputs
)

# Create agent with wrapped tools
agent = create_openai_tools_agent(llm, wrapped_tools, prompt)
executor = AgentExecutor(agent=agent, tools=wrapped_tools)

# Tool outputs are automatically compressed
result = executor.invoke({"input": "Find users who logged in yesterday"})
```

**Per-tool metrics:**

```python
from headroom.integrations import get_tool_metrics

metrics = get_tool_metrics()
print(metrics.get_summary())
# {
#   'total_invocations': 25,
#   'total_compressions': 18,
#   'total_chars_saved': 450000,
#   'by_tool': {
#     'search_database': {'invocations': 15, 'chars_saved': 320000},
#     'fetch_logs': {'invocations': 10, 'chars_saved': 130000},
#   }
# }
```

---

### 5. Streaming Metrics

Track output tokens during streaming:

```python
from headroom.integrations import StreamingMetricsTracker

tracker = StreamingMetricsTracker(model="gpt-4o")

for chunk in llm.stream("Write a poem about coding"):
    tracker.add_chunk(chunk)
    print(chunk.content, end="", flush=True)

metrics = tracker.finish()
print(f"\nOutput tokens: {metrics.output_tokens}")
print(f"Duration: {metrics.duration_ms:.0f}ms")
```

**Context manager style:**

```python
from headroom.integrations import StreamingMetricsCallback

with StreamingMetricsCallback(model="gpt-4o") as tracker:
    for chunk in llm.stream(messages):
        tracker.add_chunk(chunk)
        print(chunk.content, end="")

print(f"Metrics: {tracker.metrics}")
```

---

### 6. LangSmith Integration

Add Headroom metrics to LangSmith traces:

```python
from headroom.integrations import HeadroomLangSmithCallbackHandler

# Create callback handler
langsmith_handler = HeadroomLangSmithCallbackHandler()

# Use with your LLM
llm = HeadroomChatModel(
    ChatOpenAI(model="gpt-4o"),
    callbacks=[langsmith_handler],
)

# After calls, metrics appear in LangSmith traces:
# - headroom.tokens_before
# - headroom.tokens_after
# - headroom.tokens_saved
# - headroom.compression_ratio
```

---

## Real-World Examples

### Example 1: LangGraph ReAct Agent

The ReAct pattern is the most common agent architecture. Here's how to optimize it:

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from headroom.integrations import HeadroomChatModel, wrap_tools_with_headroom

# Define tools that return large outputs
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulating large search results
    return json.dumps({
        "results": [
            {"title": f"Result {i}", "snippet": "..." * 100, "url": f"https://..."}
            for i in range(100)
        ],
        "total": 1000,
    })

@tool
def query_database(sql: str) -> str:
    """Execute SQL query."""
    return json.dumps({
        "rows": [{"id": i, "data": "..." * 50} for i in range(500)],
        "total": 500,
    })

# Wrap model with Headroom
llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))

# Wrap tools with compression
tools = wrap_tools_with_headroom([search_web, query_database])

# Create ReAct agent
agent = create_react_agent(llm, tools)

# Run - tool outputs are automatically compressed between iterations
result = agent.invoke({
    "messages": [("user", "Find all users who signed up last week and their activity")]
})

# Check savings
print(f"Tokens saved: {llm.get_metrics()['tokens_saved']}")
```

**Without Headroom**: Each tool call adds 10-50K tokens to context.
**With Headroom**: Tool outputs compressed to 1-2K tokens, agent runs faster and cheaper.

---

### Example 1b: LangGraph Custom Graph with compress_tool_messages Node

If you're building a custom LangGraph `StateGraph` (instead of using `create_react_agent`),
you can insert a compression node between tools and the agent. This compresses all
`ToolMessage` content in the graph state before the LLM sees it.

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from headroom.integrations.langchain import create_compress_tool_messages_node

# Define your agent and tools nodes
def agent_node(state: MessagesState):
    llm = ChatOpenAI(model="gpt-4o")
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def tools_node(state: MessagesState):
    # Your tool execution logic here
    ...

# Build the graph with a compression step
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tools_node)
graph.add_node("compress", create_compress_tool_messages_node(
    min_tokens_to_compress=100,  # Only compress outputs > ~100 tokens
))

# Wire: tools -> compress -> agent (instead of tools -> agent directly)
graph.add_edge(START, "agent")
graph.add_edge("tools", "compress")
graph.add_edge("compress", "agent")
# ... add conditional edges from agent to tools/END as needed

app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="Find sales data")]})
```

You can also use `compress_tool_messages` directly as a standalone function:

```python
from headroom.integrations.langchain import compress_tool_messages

# Compress ToolMessages in any list of LangChain messages
result = compress_tool_messages(messages, min_tokens_to_compress=100)
compressed_messages = result.messages
print(f"Saved {result.total_tokens_saved} tokens across {result.messages_compressed} messages")
```

---

### Example 2: RAG Pipeline with Document Filtering

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from headroom.integrations import HeadroomChatModel, HeadroomDocumentCompressor

# Setup vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings)

# High-recall retriever (get many candidates)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

# Headroom compressor for precision
compressor = HeadroomDocumentCompressor(
    max_documents=5,       # Keep only top 5
    min_relevance=0.4,     # Must be 40%+ relevant
    prefer_diverse=True,   # Avoid redundant docs
)

# Combine into compression retriever
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

# Wrap LLM
llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# Query - retrieves 50 docs, uses best 5
result = qa_chain.invoke({"query": "How do I configure authentication?"})
print(f"Answer: {result['result']}")
print(f"Sources: {len(result['source_documents'])} docs")
```

**Impact**:
- Without filtering: 50 docs × ~500 tokens = 25K context tokens
- With Headroom: 5 docs × ~500 tokens = 2.5K context tokens (90% reduction)

---

### Example 3: Conversational Agent with Memory

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import ConversationChain
from headroom.integrations import HeadroomChatModel, HeadroomChatMessageHistory

# Wrap LLM
llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))

# Wrap memory with auto-compression
base_history = ChatMessageHistory()
compressed_history = HeadroomChatMessageHistory(
    base_history,
    compress_threshold_tokens=8000,  # Compress when over 8K
    keep_recent_turns=10,            # Always keep last 10 turns
)

memory = ConversationBufferMemory(
    chat_memory=compressed_history,
    return_messages=True,
)

# Create conversation chain
chain = ConversationChain(llm=llm, memory=memory)

# Long conversation - memory auto-compresses
for i in range(100):
    response = chain.invoke({"input": f"Tell me about topic {i}"})
    print(f"Turn {i}: {len(response['response'])} chars")

# Check memory stats
print(compressed_history.get_compression_stats())
# {'compression_count': 8, 'total_tokens_saved': 45000}
```

**Impact**: Without compression, 100-turn conversation = 100K+ tokens. With HeadroomChatMessageHistory, it stays under 8K tokens while preserving recent context.

---

### Example 4: Multi-Tool Research Agent

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from headroom.integrations import (
    HeadroomChatModel,
    wrap_tools_with_headroom,
    get_tool_metrics,
    reset_tool_metrics,
)

@tool
def search_arxiv(query: str) -> str:
    """Search arXiv for papers."""
    return json.dumps({"papers": [{"title": f"Paper {i}", "abstract": "..." * 200} for i in range(50)]})

@tool
def search_github(query: str) -> str:
    """Search GitHub repositories."""
    return json.dumps({"repos": [{"name": f"repo-{i}", "description": "..." * 100, "stars": i * 100} for i in range(100)]})

@tool
def fetch_documentation(url: str) -> str:
    """Fetch documentation from URL."""
    return "..." * 5000  # Large doc content

# Wrap everything
llm = HeadroomChatModel(ChatOpenAI(model="gpt-4o"))
tools = wrap_tools_with_headroom([search_arxiv, search_github, fetch_documentation])

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant. Use tools to gather information."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Reset metrics for this session
reset_tool_metrics()

# Run complex research task
result = executor.invoke({
    "input": "Research the latest advances in LLM context compression and find relevant GitHub projects"
})

# Check per-tool metrics
metrics = get_tool_metrics().get_summary()
print(f"Total chars saved: {metrics['total_chars_saved']:,}")
print(f"Per-tool breakdown: {metrics['by_tool']}")
```

---

## Configuration Options

### HeadroomChatModel

```python
HeadroomChatModel(
    wrapped_model,                     # Any LangChain BaseChatModel
    headroom_config=HeadroomConfig(),  # Headroom configuration
    auto_detect_provider=True,         # Auto-detect from wrapped model
)
```

### HeadroomChatMessageHistory

```python
HeadroomChatMessageHistory(
    base_history,                      # Any BaseChatMessageHistory
    compress_threshold_tokens=4000,    # Token threshold for compression
    keep_recent_turns=5,               # Minimum turns to preserve
    model="gpt-4o",                    # Model for token counting
)
```

### HeadroomDocumentCompressor

```python
HeadroomDocumentCompressor(
    max_documents=10,                  # Maximum docs to return
    min_relevance=0.0,                 # Minimum relevance score (0-1)
    prefer_diverse=False,              # Use MMR for diversity
)
```

### wrap_tools_with_headroom

```python
wrap_tools_with_headroom(
    tools,                             # List of LangChain tools
    min_chars_to_compress=1000,        # Minimum output size
    smart_crusher_config=None,         # SmartCrusher configuration
)
```

---

## Import Reference

```python
from headroom.integrations import (
    # Chat Model
    HeadroomChatModel,

    # Memory
    HeadroomChatMessageHistory,

    # Retrievers
    HeadroomDocumentCompressor,

    # Agents
    HeadroomToolWrapper,
    wrap_tools_with_headroom,
    get_tool_metrics,
    reset_tool_metrics,

    # Streaming
    StreamingMetricsTracker,
    StreamingMetricsCallback,
    track_streaming_response,

    # LangSmith
    HeadroomLangSmithCallbackHandler,

    # Provider Detection
    detect_provider,
    get_headroom_provider,
)

# Or import from subpackage directly
from headroom.integrations.langchain import HeadroomChatModel
from headroom.integrations.langchain.memory import HeadroomChatMessageHistory
```

---

## Troubleshooting

### LangChain not detected

```python
from headroom.integrations import langchain_available

if not langchain_available():
    print("Install with: pip install headroom-ai[langchain]")
```

### Provider detection failing

```python
# Force a specific provider
from headroom.providers import AnthropicProvider

llm = HeadroomChatModel(
    ChatAnthropic(model="claude-3-5-sonnet-20241022"),
    auto_detect_provider=False,
)
llm._provider = AnthropicProvider()
```

### Memory not compressing

Check that your message count exceeds the threshold:

```python
history = HeadroomChatMessageHistory(
    base_history,
    compress_threshold_tokens=1000,  # Lower threshold
    keep_recent_turns=2,             # Fewer preserved turns
)
```

---

## Performance Tips

1. **Use tool wrapping for agents** - Agents with tools benefit most from compression
2. **Set appropriate thresholds** - Don't compress small conversations
3. **Enable diversity for RAG** - `prefer_diverse=True` improves answer quality
4. **Monitor with LangSmith** - Use the callback handler to track savings over time
5. **Batch similar requests** - Provider caching works better with stable prefixes
