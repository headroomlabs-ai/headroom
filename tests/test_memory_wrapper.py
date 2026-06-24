from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from headroom.memory.config import EmbedderBackend
from headroom.memory.wrapper import MemoryWrapper, _MemoryAPI, with_memory


class FakeMemory:
    def __init__(self) -> None:
        self.search_results: list[object] = []
        self.add_calls: list[dict[str, object]] = []
        self.query_results: list[object] = []
        self.clear_result = 0

    async def search(self, **kwargs):  # noqa: ANN003
        self.last_search = kwargs
        return self.search_results

    async def add(self, **kwargs):  # noqa: ANN003
        self.add_calls.append(kwargs)
        return SimpleNamespace(id=f"mem-{len(self.add_calls)}", **kwargs)

    async def query(self, filter_value):  # noqa: ANN001, ANN201
        self.last_filter = filter_value
        return self.query_results

    async def clear_scope(self, **kwargs):  # noqa: ANN003
        self.last_clear = kwargs
        return self.clear_result


def make_client(content: str = "raw response") -> tuple[object, object]:
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    def create(**kwargs):  # noqa: ANN003, ANN202
        create.kwargs = kwargs
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, response


def test_memory_wrapper_lazy_initialization_and_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _response = make_client()
    fake_memory = FakeMemory()
    seen: dict[str, object] = {}

    async def fake_create(config):  # noqa: ANN001
        seen["config"] = config
        return fake_memory

    monkeypatch.setattr("headroom.memory.wrapper.HierarchicalMemory.create", fake_create)

    wrapper = MemoryWrapper(
        client,
        user_id="alice",
        db_path="memory.db",
        top_k=7,
        session_id="session-1",
        agent_id="agent-1",
        embedder_backend=EmbedderBackend.OPENAI,
        openai_api_key="sk-test",
    )

    assert wrapper.chat.completions._wrapper is wrapper
    assert wrapper._initialized is False

    api = wrapper.memory
    assert isinstance(api, _MemoryAPI)
    assert wrapper._initialized is True
    assert wrapper._memory is fake_memory
    assert seen["config"].db_path == Path("memory.db")
    assert seen["config"].embedder_backend == EmbedderBackend.OPENAI
    assert seen["config"].openai_api_key == "sk-test"

    wrapped = with_memory(client, user_id="bob", session_id="s2", agent_id="a2", top_k=3)
    assert isinstance(wrapped, MemoryWrapper)
    assert wrapped._client is client
    assert wrapped._user_id == "bob"
    assert wrapped._session_id == "s2"
    assert wrapped._agent_id == "a2"
    assert wrapped._top_k == 3


def test_inject_memories_handles_empty_and_inserts_context() -> None:
    client, _response = make_client()
    fake_memory = FakeMemory()
    wrapper = MemoryWrapper(client, user_id="alice", _memory=fake_memory)

    no_user = [{"role": "assistant", "content": "skip"}]
    assert wrapper._inject_memories(no_user) == no_user

    messages = [{"role": "user", "content": "Question?"}]
    assert wrapper._inject_memories(messages) == messages

    fake_memory.search_results = [
        SimpleNamespace(memory=SimpleNamespace(content="Prefers Python")),
        SimpleNamespace(memory=SimpleNamespace(content="Works on APIs")),
    ]
    original = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Question?"},
        {"role": "user", "content": "Follow-up"},
    ]
    injected = wrapper._inject_memories(original)

    assert original[1]["content"] == "Question?"
    assert injected[1]["content"].startswith(
        "<context>\n- Prefers Python\n- Works on APIs\n</context>\n\n"
    )
    assert injected[2]["content"] == "Follow-up"
    assert fake_memory.last_search == {
        "query": "Follow-up",
        "user_id": "alice",
        "session_id": None,
        "top_k": 5,
    }


def test_store_memories_persists_only_nonempty_content() -> None:
    client, _response = make_client()
    fake_memory = FakeMemory()
    wrapper = MemoryWrapper(
        client,
        user_id="alice",
        session_id="session-1",
        agent_id="agent-1",
        _memory=fake_memory,
    )

    wrapper._store_memories([{"content": "Remember this"}, {"content": ""}, {}])

    assert fake_memory.add_calls == [
        {
            "content": "Remember this",
            "user_id": "alice",
            "session_id": "session-1",
            "agent_id": "agent-1",
            "importance": 0.7,
        }
    ]


def test_wrapped_completions_create_injects_parses_and_stores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client, response = make_client("raw completion")
    wrapper = MemoryWrapper(client, user_id="alice", _memory=FakeMemory())
    stored: list[list[dict[str, str]]] = []

    monkeypatch.setattr(
        wrapper,
        "_inject_memories",
        lambda messages: [{"role": "user", "content": "enhanced"}],
    )
    monkeypatch.setattr(
        "headroom.memory.wrapper.inject_memory_instruction",
        lambda messages, short=True: (
            messages + [{"role": "system", "content": "memory-instruction"}]
        ),
    )
    monkeypatch.setattr(
        "headroom.memory.wrapper.parse_response_with_memory",
        lambda content: SimpleNamespace(
            content="clean response",
            memories=[{"content": "saved memory"}],
        ),
    )
    monkeypatch.setattr(wrapper, "_store_memories", lambda memories: stored.append(memories))

    result = wrapper.chat.completions.create(
        messages=[{"role": "user", "content": "hello"}], model="x"
    )

    assert result is response
    assert response.choices[0].message.content == "clean response"
    assert client.chat.completions.create.kwargs["messages"] == [
        {"role": "user", "content": "enhanced"},
        {"role": "system", "content": "memory-instruction"},
    ]
    assert stored == [[{"content": "saved memory"}]]


def test_memory_api_methods_delegate_to_underlying_memory() -> None:
    fake_memory = FakeMemory()
    memory_one = SimpleNamespace(id="m1", content="alpha")
    memory_two = SimpleNamespace(id="m2", content="beta")
    fake_memory.search_results = [
        SimpleNamespace(memory=memory_one),
        SimpleNamespace(memory=memory_two),
    ]
    fake_memory.query_results = [memory_one, memory_two]
    fake_memory.clear_result = 2

    api = _MemoryAPI(fake_memory, user_id="alice", session_id="session-1", agent_id="agent-1")

    assert api.search("alpha", top_k=3) == [memory_one, memory_two]
    added = api.add("new memory", importance=0.9)
    assert added.content == "new memory"
    assert api.get_all() == [memory_one, memory_two]
    assert api.clear() == 2
    assert api.stats() == {"total": 2}
    assert fake_memory.last_clear == {"user_id": "alice"}
