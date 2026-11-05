from __future__ import annotations

import gc
import weakref
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

from headroom.client import HeadroomClient


class _ClosableIterator:
    def __init__(
        self,
        values: list[object] | None = None,
        error: Exception | None = None,
        close_error: Exception | None = None,
    ):
        self._values = iter(values or [])
        self._error = error
        self._close_error = close_error
        self.closed = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        return next(self._values)

    def close(self):
        self.closed += 1
        if self._close_error is not None:
            raise self._close_error


class _IterableOnly:
    def __init__(self, values: list[object]):
        self._values = values
        self.closed = 0

    def __iter__(self):
        return iter(self._values)

    def close(self):
        self.closed += 1


class _RetainedIterable:
    def __init__(self):
        self.iterator = None
        self.closed = 0

    def __iter__(self):
        def generate():
            try:
                yield "chunk"
                yield "later"
            finally:
                self.closed += 1

        self.iterator = generate()
        return self.iterator


class _DistinctClosableSource:
    def __init__(
        self, owner_error: Exception | None = None, iterator_error: Exception | None = None
    ):
        self.owner_error = owner_error
        self.iterator_error = iterator_error
        self.closed = 0
        self.iterator = None

    def __iter__(self):
        self.iterator = _ClosableIterator(["chunk"], close_error=self.iterator_error)
        return self.iterator

    def close(self):
        self.closed += 1
        if self.owner_error is not None:
            raise self.owner_error


class _Owner:
    def __init__(self):
        self.saved = []
        self._storage = SimpleNamespace(save=self.saved.append)


class _ClosableSse(httpx.SyncByteStream):
    def __init__(self):
        self.closed = 0

    def __iter__(self):
        yield (
            b'data: {"id":"x","object":"chat.completion.chunk","created":1,'
            b'"model":"gpt","choices":[{"index":0,"delta":{"content":"hi"},'
            b'"finish_reason":null}]}\n\n'
        )
        yield b"data: [DONE]\n\n"

    def close(self):
        self.closed += 1


def _openai_stream(body: _ClosableSse):
    def handler(request: httpx.Request):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=body,
        )

    http_client = httpx.Client(transport=httpx.MockTransport(handler))
    client = OpenAI(api_key="test", http_client=http_client)
    stream = client.chat.completions.create(
        model="gpt-test",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    return stream, http_client


def test_wrap_stream_closes_source_before_iteration_and_saves_once():
    owner = _Owner()
    source = _ClosableIterator(["chunk"])
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    wrapped.close()
    wrapped.close()

    assert source.closed == 1
    assert owner.saved == ["metrics"]


def test_wrap_stream_closes_source_after_exhaustion_and_saves_once():
    owner = _Owner()
    source = _ClosableIterator(["chunk"])
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    assert list(wrapped) == ["chunk"]
    wrapped.close()

    assert source.closed == 1
    assert owner.saved == ["metrics"]


def test_wrap_stream_does_not_iterate_after_close():
    owner = _Owner()
    source = _ClosableIterator(["chunk"])
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    wrapped.close()

    with pytest.raises(StopIteration):
        next(wrapped)
    assert owner.saved == ["metrics"]


def test_wrap_stream_closes_source_on_error_and_saves_once():
    owner = _Owner()
    source = _ClosableIterator(error=RuntimeError("boom"))
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    with pytest.raises(RuntimeError, match="boom"):
        next(wrapped)

    assert source.closed == 1
    assert owner.saved == ["metrics"]


def test_wrap_stream_supports_iterators_without_close():
    owner = _Owner()
    wrapped = HeadroomClient._wrap_stream(owner, iter(["chunk"]), "metrics")

    assert list(wrapped) == ["chunk"]
    assert owner.saved == ["metrics"]


def test_wrap_stream_supports_iterable_only_sources():
    owner = _Owner()
    source = _IterableOnly(["chunk"])
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    assert list(wrapped) == ["chunk"]
    assert source.closed == 1
    assert owner.saved == ["metrics"]


def test_wrap_stream_preserves_iteration_error_when_close_fails():
    owner = _Owner()
    source = _ClosableIterator(
        error=RuntimeError("iteration"),
        close_error=RuntimeError("close"),
    )
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    with pytest.raises(RuntimeError, match="iteration"):
        next(wrapped)

    assert source.closed == 1
    assert owner.saved == ["metrics"]


def test_wrap_stream_closes_retained_iterator_when_source_has_no_close():
    owner = _Owner()
    source = _RetainedIterable()
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    assert next(wrapped) == "chunk"
    wrapped.close()

    assert source.closed == 1
    assert owner.saved == ["metrics"]


def test_wrap_stream_closes_distinct_source_and_iterator():
    owner = _Owner()
    source = _DistinctClosableSource()
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    assert next(wrapped) == "chunk"
    wrapped.close()

    assert source.closed == 1
    assert source.iterator.closed == 1
    assert owner.saved == ["metrics"]


def test_wrap_stream_closes_both_owners_when_each_close_fails():
    owner = _Owner()
    source = _DistinctClosableSource(
        owner_error=RuntimeError("owner close"),
        iterator_error=RuntimeError("iterator close"),
    )
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    assert next(wrapped) == "chunk"
    wrapped.close()
    wrapped.close()

    assert source.closed == 1
    assert source.iterator.closed == 1
    assert owner.saved == ["metrics"]


def test_wrap_stream_finalizes_started_abandoned_stream():
    owner = _Owner()
    source = _ClosableIterator(["chunk"])
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    next(wrapped)
    del wrapped
    gc.collect()

    assert source.closed == 1
    assert owner.saved == ["metrics"]


def test_closed_stream_releases_source_and_owner():
    owner = _Owner()
    source = _ClosableIterator(["chunk"])
    owner_ref, source_ref = weakref.ref(owner), weakref.ref(source)
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    wrapped.close()
    del owner, source
    gc.collect()

    assert owner_ref() is None
    assert source_ref() is None


def test_wrap_stream_context_manager_closes_on_consumer_error():
    owner = _Owner()
    source = _ClosableIterator(["chunk"])

    with pytest.raises(RuntimeError, match="consumer"):
        with HeadroomClient._wrap_stream(owner, source, "metrics") as wrapped:
            assert next(wrapped) == "chunk"
            raise RuntimeError("consumer")

    assert source.closed == 1
    assert owner.saved == ["metrics"]


def test_wrap_real_openai_stream_closes_before_first_chunk():
    body = _ClosableSse()
    source, http_client = _openai_stream(body)
    owner = _Owner()
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    wrapped.close()
    http_client.close()

    assert body.closed == 1
    assert owner.saved == ["metrics"]


def test_wrap_real_openai_stream_closes_after_first_chunk():
    body = _ClosableSse()
    source, http_client = _openai_stream(body)
    owner = _Owner()
    wrapped = HeadroomClient._wrap_stream(owner, source, "metrics")

    next(wrapped)
    wrapped.close()
    http_client.close()

    assert body.closed == 1
    assert owner.saved == ["metrics"]
