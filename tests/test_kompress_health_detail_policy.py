"""The public kompress health detail must stay inside a closed vocabulary."""

import pytest

from headroom.proxy.kompress_health_detail_policy import (
    DEFAULT_KOMPRESS_HEALTH_DETAIL,
    KOMPRESS_HEALTH_DETAILS,
    public_detail,
)


@pytest.mark.parametrize("value", sorted(KOMPRESS_HEALTH_DETAILS))
def test_known_details_pass_through(value):
    assert public_detail(value) == value


@pytest.mark.parametrize(
    "value",
    [
        "warm failed: /home/opt/models/modernbert.onnx not readable",
        "OSError: cannot open shared object /usr/lib/libonnxruntime.so",
        "HTTPError 401 for https://internal.example/models?token=abc123",
        "",
        "   ",
        None,
        123,
    ],
)
def test_unknown_details_collapse(value):
    assert public_detail(value) == DEFAULT_KOMPRESS_HEALTH_DETAIL


def test_details_are_normalized():
    assert public_detail("  Model Not Cached  ") == "model not cached"


def test_default_is_itself_a_known_detail():
    assert DEFAULT_KOMPRESS_HEALTH_DETAIL in KOMPRESS_HEALTH_DETAILS
