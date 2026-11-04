from types import SimpleNamespace

import pytest

from headroom.transforms.content_router import ContentRouter, ContentRouterConfig
from headroom.transforms.dense_line_elider import elide_dense_lines
from headroom.transforms.recursive_json import json_document_spans, scan_json_documents


@pytest.mark.parametrize("wrapper", ["[response: {}]", "{{response: {}}}"])
def test_nested_record_document_survives_non_json_wrapper(monkeypatch, wrapper):
    document = '{"items":[{"id":1},{"id":2}]}'
    payload = wrapper.format(document)
    calls = []

    class Kompress:
        def is_ready(self):
            return True

        def compress(self, content, **kwargs):
            calls.append(content)
            return SimpleNamespace(compressed="records deleted", compressed_tokens=2)

    router = ContentRouter(ContentRouterConfig())
    monkeypatch.setattr(router, "_get_kompress", lambda: Kompress())
    out, _ = router._try_ml_compressor(payload, context="")
    assert out == payload
    assert calls == []
    assert [payload[a:b] for a, b in json_document_spans(payload)] == [document]


def test_dense_prefixed_json_retains_its_value():
    payload = 'Tool result: {"token":"' + "x" * 3500 + '"}'
    assert elide_dense_lines(payload) == (payload, 0)


def test_invalid_wrapper_parsing_has_a_linear_character_budget(monkeypatch):
    import headroom.transforms.recursive_json as recursive_json

    text = "[" * 500 + "response: " + '{"items":[{"id":1},{"id":2}]}' + "]" * 500
    parsed_chars = 0
    real_loads = recursive_json.json.loads

    def count_loads(value, *args, **kwargs):
        nonlocal parsed_chars
        parsed_chars += len(value)
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr(recursive_json.json, "loads", count_loads)
    _spans, complete = scan_json_documents(text)
    assert not complete
    assert parsed_chars <= 4 * len(text) + 4096
